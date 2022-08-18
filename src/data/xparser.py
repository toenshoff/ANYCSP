# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 17:16:42 2021

@author: Jakob Lindner

requires python 3.8 or higher
"""

import xml.etree.ElementTree as ET
import functools
import numpy as np
import re
import torch
from src.csp import csp_data
from src.data import tuple_generator
import copy
import operator as op
from src.data.intension import intension_to_linear



class XParser(object):
    """
    class for XML Parser for XCSP3 instances
    assumes well formed xcsp3 instances
    """
    
    def __init__(self, path, num_samples = 32):
        """
        Constructor for XParser objects

        Parameters
        ----------
        path : location of the file
        num_samples : passed to network. The default is 32.

        Raises
        ------
        Exception
            instance is not properly formated
        """
        self.path = path
        self.num_var = 0
        self.var_to_num = {} 
        self.domain = np.empty(shape=0, dtype = np.int64)
        self.constraints = [] 
        self.groups = []
        self.tree = ET.parse(self.path)
        self.array_dimensions = {}
        self.num_samples = num_samples
        self.pass_domains = False
        self.instantiation = []

        if self.tree.getroot().attrib['format'] != 'XCSP3':
            raise Exception("format of the document can not be parsed")
            
        
    def plain_extension(self):
        """
        utility function to check if an instance contains only basic
        extension type constraints
        used to speed up parsing by ignoring most potential edge cases
        
        Returns
        -------
        bool
        """
        if self.instantiation != []:
            return False
        for const in self.constraints:
            if type(const) != Extension:
                return False
        for group in self.groups:
            if type(group.const) != Extension:
                return False
        for dom in self.domain:
            if dom != self.domain[0]:
                return False
        return True
        
        
    """
    ---------------------------------------------------------------------------
    Methods to read in variables
    ---------------------------------------------------------------------------
    """
    
    def get_var_domain(self):
        """
        update variables, numberOfVar and their corresponding domains 
        by the given XML tree of the XParser object
        
        supports:
            - integer variables
            - arrays of integer variables

        Raises
        ------
        Exception
            mixed domains for arrays not implemented

        """
        # clear old values
        self.tree = ET.parse(self.path)
        self.var_to_num = {} 
        self.domain = np.empty(shape = 0, dtype=np.int64)
        self.num_var = 0
        self.array_dimensions = {}
        self.constraints = []
        self.groups = []
        
        for var in self.tree.getroot().findall('variables/var'):
            self.var_to_num[var.attrib['id']] = self.num_var
            if var.get('as', default = False):
                if not self.domain[self.var_to_num[var.get('as')]]:
                    raise Exception("can't find identifier")
                self.domain = np.append(self.domain, self.domain[self.var_to_num[var.get('as')]]) 
            else:
                self.domain = np.append(self.domain, XParser.extract_num_dom(var.text.lstrip().rstrip()))
            self.num_var += 1
       
        for array in self.tree.getroot().findall('variables/array'):
            dimensions = array.attrib['size'].replace("]", "").split("[")
            dimensions = [int(x) for x in dimensions[1:]]
            self.array_dimensions[array.attrib['id']] = dimensions
            size = functools.reduce(lambda x,y: x*y, dimensions, 1)
            for var in range(size):
                var_name = array.attrib['id']
                for dim in range(len(dimensions)):
                    var_name += "[" + str((int((var % functools.reduce(lambda x,y: x*y, dimensions[dim:], 1)) / functools.reduce(lambda x,y: x*y, dimensions[dim+1:], 1) ))) +"]"
                self.var_to_num[var_name] = self.num_var
                self.num_var += 1
            
            if array.find('domain') != None:
                raise Exception("Mixed domains are yet to implement")
            else:
                self.domain = np.append(self.domain, np.repeat(XParser.extract_num_dom(array.text.lstrip().rstrip()), size))
        
        self.read_instantiation()
    
    
    @staticmethod
    def extract_num_dom(text):
        """
        converts the text representation of a domain into a dictionary mapping 
        every value to increasing integer values
        e.g. 1..7 to {1:0, 2:1, 3:2, 4:3, 5:4, 6:4, 7:6}

        Parameters
        ----------
        text : string of the domain

        Returns
        -------
        var_dom : dictionary mapping domain values to consecutive integers
        """
        var_dom = {}
        num_val = 0
        for elem in text.split(' '):
            if '..' in elem: 
                limits= elem.split('..')
                for val in range(int(limits[0]), int(limits[1])+1):
                    var_dom[val] = num_val
                    num_val += 1
            else:
                var_dom[int(elem)] = num_val
                num_val += 1
        return var_dom
    
    
    """
    ---------------------------------------------------------------------------
    Read in Constraints
    ---------------------------------------------------------------------------
    """
    
    def get_constraints(self, root):
        """
        update the constraints of the object by reading from the tree 
        recursivly for blocks
        
        supports:
            - extension type constraints
            - allDiffrent type constraints
            - sum type constraints
            - groups of constraints
            - blocks of constraints

        Parameters
        ----------
        root : root node to start reading constrints from

        Raises
        ------
        Exception
            for non-implemented types of constraints
        """
        for child in root:
            if child.tag == 'block':
                self.get_constraints(child)
            elif child.tag == 'extension':
                const = self.get_extension(child)
                if const:
                    self.constraints.append(const)
            elif child.tag == 'allDifferent':
                res = self.get_allDifferent(child)
                if type(res) == Comparison:
                    self.constraints.append(res)
                elif type(res) == list:
                    self.constraints += res
                else:
                    continue
            elif child.tag == 'sum':
                self.constraints.append(self.get_linear_equation(child))
            elif child.tag == 'intension':
                self.constraints.append(Intension(child.find('intension').text.lstrip().rstrip()))
            elif child.tag == 'group':
                arguments = []
                if child.find('extension') != None:
                    param = self.get_extension(child.find('extension'))
                elif child.find('allDifferent') != None:
                    param = self.get_allDifferent(child.find('allDifferent'))
                elif child.find('sum') != None:
                    param = self.get_linear_equation(child.find('sum'))    
                elif child.find('intension') != None:
                    for arg in child.findall('args'):
                        arg = self.expand_notation(arg.text)
                        intension_text = child.find('intension').text.lstrip().rstrip()
                        self.constraints.append(Intension(intension_text, arg))
                    continue
                else:
                    ctype = child.find("*").tag
                    raise Exception("groups of constraint type '" + str(ctype) + "' not yet implemented")
                if not param:
                    continue
                for arg in child.findall('args'):
                    arguments.append(self.expand_notation(arg.text))
                self.groups.append(Group(param, arguments))
            elif child.tag == 'instantiation':
                continue
            else:
                raise Exception("constraint type '" + str(child.tag) + "' not implemented")
    
    
    def get_extension(self, extRoot):
        """
        reads an extension from the given node extRoot and 
        returns an new object of type Extensions

        Parameters
        ----------
        extRoot : 
            node to read extension from

        Returns
        -------
        ext : 
            Extension type object
        """
        negate = True
        var = self.expand_notation(extRoot.find('list').text)
        if extRoot.find('supports') != None:
            supports = XParser.to_tuple_list(extRoot.find('supports').text)
            negate = False
        else:
            supports = []
        if extRoot.find('conflicts') != None:
            conflicts = XParser.to_tuple_list(extRoot.find('conflicts').text)
        else:
            conflicts = []
        if supports == [] and conflicts == []:
            return None
        return Extension(var, supports, conflicts, negate)
    
    
    @staticmethod
    def to_tuple_list(text):
        """
        splits string of tuples into a list
        expand compact notation
        helper for get_extension

        Parameters
        ----------
        text : 
            string of tuples

        Returns
        -------
        TYPE
            tuples as list
        """
        text = text.lstrip().rstrip()
        if re.match(r"^(-{0,1}\d+ )", text):
            items = text.split(' ')
            res = []
            for i in items: 
                if match := re.match(r"(-{0,1}\d+)\.\.(-{0,1}\d+)", i):
                    start, stop = match.groups()
                    for val in range(int(start), int(stop) + 1):
                        res.append([val])
                else: 
                    res.append([int(i)])
            return res
        tuple_list = text.replace('(','').split(")")
        tuple_list = [[int(x) for x in item.split(",")] for item in tuple_list[:-1]]
        return tuple_list
    
    
    def get_allDifferent(self, root):
        """
        reads an allDiffrent type constraint

        Parameters
        ----------
        root : 
            node to read constraint from

        Raises
        ------
        Exception
            constrint type not implemented

        Returns
        -------
        Comparison
            Comparison type object

        """
        if root.find('matrix') != None:
            tuples = self.expand_matrix_notation(root.find('matrix').text)
            res = [Comparison(var, []) for var in tuples]
            for dim in range(len(tuples[0])):
                res.append(Comparison([tuples[i][dim] for i in range(len(tuples))], []))
            return res
        elif root.find('list') != None:
            var_list = []
            for list in root.findall('list'):
                var_list.append(self.expand_notation(list.text))
            if len(var_list) == 1:    
                var = self.expand_notation(root.find('list').text)
                except_val = root.find('except').lstrip().rstrip().split(' ')
                return Comparison(var, except_val)  
            else:
                raise Exception('allDifferent lifted to lists not implemented')
        else: 
            var = self.expand_notation(root.text)
            return Comparison(var, [])    
    
    
    def get_linear_equation(self, root):
        """
        reads an sum type constraint
        caution: does not support variable coefficients or operands (TODO)

        Parameters
        ----------
        root : 
            node to read constraint from

        Returns
        -------
        LinearEquation
            new object of type LinearEquation
        """
        var = self.expand_notation(root.find('list').text)
        if root.find('coeffs') != None:
            coeffs = XParser.expand_compact_integer(root.find('coeffs').text)
        else: 
            coeffs = []
        operator, operand = root.find('condition').text.replace('(','').replace(')','').split(',')
        operator = operator.lstrip().rstrip()
        operand = int(operand)
        return LinearEquation(var, coeffs, operator, operand)
    
    
    def read_instantiation(self):
        """
        read given instantiation of the file if any
        """
        root = self.tree.getroot().find('constraints').find('instantiation')
        if not root:
            return
        if root.get('type') in ['solution','optimum']:
            return
        var = self.expand_notation(root.find('list').text)
        val = XParser.expand_compact_integer(root.find('values').text)
        for i in range(len(var)):
            if val[i] == '*':
                continue
            self.domain[self.var_to_num[var[i]]] = {int(val[i]) : 0}
        self.instantiation = var
            
    
    def preprocessing(self):
        """
        Call preprocessing method for all Constraints and Groups

        Raises
        ------
        Exception
            Instance is unsatisfiable
        """
        for const in self.constraints:
            const.preprocessing(self)
        new_groups = []
        for group in self.groups:
            group.preprocessing(self)
            if group.arguments != []:
                new_groups.append(group)
        self.groups = new_groups

        # check satisfiablity
        for dom in self.domain:
            if len(dom) == 0:
                raise Exception('instance is unsatisfialbe after preprocessing')
        
        
    """
    ---------------------------------------------------------------------------
    Methods to form tensors representing constraints
    ---------------------------------------------------------------------------
    """
    
    def collect_data(self, max_tuples_per_eq, implicid_data = True,  subsampl = True, max_num_tuples = 10000000000):
        """
        Main method to convert all data and merge it to consecutive representations

        Parameters
        ----------
        max_tuples_per_eq : 
            number of tuples for which an linear equation is converted to 
            expicit tuple representation
        implicid_data : boolean, optional
            set to False to force tuple representations if availible. The default is True.
        subsampl : boolean, optional
            allow subsampling. The default is True.
        max_num_tuples : integer, optional
            number of tuples to stop the parsing. The default is 10000000000.

        Raises
        ------
        Exception
            too many tuples

        Returns
        -------
        pos : 
            possitive explicit constraint data
        neg : 
            negative explicit constraint data
        uni : 
            uniform constraint data
        lin : 
            linear constraint data
        allDiff : 
            allDifferent constraint data
        """
        pos = Parser_constraint_data("pos", False)
        neg = Parser_constraint_data("neg", True)
        uni = []
        lin = []
        allDiff = []
        num_tuples = 0
       
        if implicid_data:
            self.pass_domains = True
        
        new_groups = []
        for group in self.groups:
            new_groups = new_groups + group.split_group()
        self.groups = new_groups  

        for group in self.groups:
            data = group.get_data(self, implicid_data, max_tuples_per_eq)
            if type(data) == Parser_uniform_data:
                num_tuples += data.var_idx.shape[0] * data.var_idx.shape[1]
                merge = False
                for uniform in uni: 
                    if uniform.name == data.name:
                        uniform.append_data(data)
                        merge = True
                if not merge:
                    uni.append(data)
            elif type(data) == Parser_constraint_data:
                num_tuples += data.num_tuple
                if data.negate:
                    neg.append_data(data)
                else:
                    pos.append_data(data)
            elif type(data) == Parser_linear_data:
                merge = False
                for eq in lin: 
                    if eq.name == data.name:
                        eq.append_data(data)
                        merge = True
                if not merge: 
                    lin.append(data)
            elif type(data) == Parser_allDifferent_data:
                merge = False
                for diff in allDiff: 
                    if diff.name == data.name:
                        diff.append_data(data)
                        merge = True
                if not merge:
                    allDiff.append(data)
            else:
                raise Exception()
            if num_tuples >= max_num_tuples:
                raise Exception("interruped parsing, to many tuples")
        
        for const in self.constraints:
            if implicid_data and type(const) == LinearEquation:
                data = const.implicit_data(self, max_num_tuples = max_tuples_per_eq)
            elif implicid_data and type(const) == Comparison:
                data = const.implicit_data(self)
            elif type(const) == Intension: 
                data = const.to_linear(self)
            else:
                data = const.to_tuple(self)
            if type(data) == Parser_constraint_data:
                ind_tuple_count = int(data.num_tuple / data.num_const)
                num_tuples += data.num_tuple * data.num_const
                if num_tuples >= max_num_tuples:
                    raise Exception("interrupt parsing, to many tuples")
                if subsampl and ind_tuple_count >= self.num_samples:
                    data_uni = data.to_uniform(self.num_samples)
                    merge = False
                    for uniform in uni: 
                        if uniform.name == data_uni.name:
                            uniform.append_data(data_uni)
                            merge = True
                    if not merge:
                        uni.append(data_uni)
                else: 
                    if data.negate:
                        neg.append_data(data)
                    else:
                        pos.append_data(data)
            elif type(data) == Parser_linear_data:
                merge = False
                for eq in lin: 
                    if eq.name == data.name:
                        eq.append_data(data)
                        merge = True
                if not merge: 
                    lin.append(data)
            elif type(data) == Parser_allDifferent_data:
                merge = False
                for diff in allDiff: 
                    if diff.name == data.name:
                        diff.append_data(data)
                        merge = True
                if not merge:
                    allDiff.append(data) 
            else:
                raise Exception()
    
        return pos, neg, uni, lin, allDiff
        

    """
    ---------------------------------------------------------------------------
    Utility for XML notation
    ---------------------------------------------------------------------------
    """
    @staticmethod
    def expand_compact_integer(text):
        """
        expands compact form of integer sequences 
        such as used in in elements
        <values> of <instantiation> and <coeffs> of <sum>
        form 'vxk' is translated to k consecutive occurences of v
        returns numpy array of integers

        Parameters
        ----------
        text : 
            text that may contain compact integer forms

        Raises
        ------
        Exception
            unknown text

        Returns
        -------
            numpy array of integers
        """
        text = text.lstrip().rstrip()
        if re.fullmatch("%...", text):
            return [text]
        items = text.split(' ')
        for i in range(len(items)): 
            item = items[i]
            if match := re.match(r"(-{0,1}\d+)x(\d+)", item):
                value, count = match.groups()
                items[i] = [int(value)] * int(count)
            elif re.match(r"-{0,1}\d+", item):
                items[i] = int(item)
            elif re.match(r"\%(\d+)", item):
                continue
            elif re.match("%...", item):
                continue
            elif re.match("\*", items):
                continue
            else: 
                raise Exception("expected integer notation, found: ", item)
        return list(np.hstack(items))
                

    @staticmethod
    def replace_var(var_list, args):
        """
        replaces "%i" notation by args
                 %... notation by all arguments

        Parameters
        ----------
        var_list : 
            list of variables containing replacment syntax
        args : 
            list of arguments to replace syntax

        Returns
        -------
        List of variables with replacments
        """
        if '%...' in var_list[0]:
            return args
        rep_list = var_list.copy()
        for j in range(len(var_list)):
            for i in reversed(range(len(args))):
                rep_list[j] = rep_list[j].replace('%' + str(i), args[i])
        return rep_list 
     
        
    @staticmethod
    def replace_coeff(coeffs, args):
        """
        replaces "%i" notation by args
                 %... notation by all arguments

        Parameters
        ----------
        coeffs : 
            list of coefficients containing replacment syntax
        args : 
            list of arguments to replace syntax

        Returns
        -------
        List of coefficients with replacments
        """
        if len(coeffs) == 0:
            return []
        if type(coeffs[0]) == str and '%...' in coeffs[0]:
            return [int(arg) for arg in args]
        rep_list = coeffs.copy()
        for j in range(len(coeffs)):
            if type(rep_list[j])==str:
                for i in reversed(range(len(args))):
                    rep_list[j] = rep_list[j].replace('%' + str(i), int(args[i]))
        return rep_list 
    
    
    def expand_notation(self, text):
        """
        expands notations "[]" and ".." but not "..."

        Parameters
        ----------
        text : 
            string

        Returns
        -------
        result : 
            list of ranges after expansion
        """
        split = text.lstrip().rstrip().split(' ')
        result = []
        for short in split:
            name = short.split('[')[0]
            range_list = re.findall(r"(\[)([\d\.]*)(\])", short)
            expansion = [name]
            for i in range(len(range_list)):
                exp = range_list[i][1]
                if exp == '':
                    range_list[i] = range(self.array_dimensions[name][i])
                elif match := re.match(r"(\d+)\.\.(\d+)", exp):
                    start, stop = match.groups()
                    range_list[i] = range(int(start), int(stop)+1)
                elif re.match(r"\d+", exp):
                    range_list[i] = range(int(exp), int(exp)+1)
                else:
                    raise Exception("notation expansion failed for", short)
            for dim in range_list:
                expansion = [part+"[%d]"%i for part in expansion for i in dim]
            result += expansion
        return result
    
    
    def expand_matrix_notation(self, text):
        """
        expands notations "[]" and ".." but not "..." for matrix notations

        Parameters
        ----------
        text : 
            string

        Returns
        -------
        result : 
            list of ranges after expansion
        """
        split = text.lstrip().rstrip().split(' ')
        tuples = []
        for short in split:
            name = short.split('[')[0]
            range_list = re.findall(r"(\[)([\d\.]*)(\])", short)
            for i in range(len(range_list)):
                exp = range_list[i][1]
                if exp == '':
                    range_list[i] = range(self.array_dimensions[name][i])
                elif match := re.match(r"(\d+)\.\.(\d+)", exp):
                    start, stop = match.groups()
                    range_list[i] = range(int(start), int(stop) + 1)
                elif re.match(r"\d+", exp):
                    range_list[i] = range(int(exp), int(exp) + 1)
                else:
                    raise Exception("notation expansion failed for", short)
            for i in range_list[0]:
                tuples.append([name + '[%d]'%i + '[%d]'%j for j in range_list[1]])
        return tuples


    @staticmethod
    def to_3d_array(tuple_count, array):
        """
        reshape array such that it is in form for uniform data

        Parameters
        ----------
        tuple_count : 
            number of tuples contained in the array
        array : 
            data to transform

        Returns
        -------
        numpy 3d array
        """
        arity = int(len(array)/tuple_count)
        array = np.reshape(array, (tuple_count, arity))
        return np.expand_dims(array, axis = 0)


    @staticmethod
    def to_stacked_3d_array(num_const, ind_tuple_count, array):
        """
        reshape array such that it is in form for uniform data for multiple constraints at once

        Parameters
        ----------
        tuple_count : 
            number of tuples contained in the array
        array : 
            data to transform

        Returns
        -------
        numpy 3d array
        """
        length = int(len(array)/num_const)
        return np.vstack([XParser.to_3d_array(ind_tuple_count, array[x*length: (x+1) * length]) for x in range(num_const)])

    
    def tuple_in_domain(self, var, tuple):
        """
        Parameters
        ----------
        var : 
            list of vairables
        tuple : 
            list of tuples

        Returns
        -------
        bool
            True if all tuples are in teh respective domains of the variables
        """
        for i in range(len(tuple)):
            if tuple[i] not in self.domain[var[i]].keys():
                return False
        return True
    
    
    """
    ---------------------------------------------------------------------------
    Functions for Interaction
    ---------------------------------------------------------------------------
    """   
    
    def to_CSP_data(self, implicid_data = True, subsampl = True, checkDomains = False, max_tuples_per_eq = 32):
        """
        Main function to convert data to csp_data object, returns new csp_data object

        Parameters
        ----------
        implicid_data : boolean, optional
            set to False to force tuple representations if availible. The default is True.
        subsampl : TYPE, optional
            allow subsampling. The default is True.
        checkDomains : boolean, optional
            check if domains match for different variables of a constraint. The default is False.
        max_tuples_per_eq : Intger, optional
            number of tuples for which an linear equation is converted to 
            expicit tuple representation. The default is 32.

        Returns
        -------
        csp : 
            csp_data object containing all data
        """
        self.get_var_domain()
        self.get_constraints(self.tree.getroot().find('constraints'))
        domain_size = torch.LongTensor([len(dom) for dom in self.domain])
        if implicid_data:
            domain = torch.LongTensor(np.hstack([[x for x in dom.keys()] for dom in self.domain]))
        else:
            domain = None
        if self.instantiation != []:
            self.preprocessing()
        csp = csp_data.CSP_Data(self.num_var, domain_size, domain, path = self.path)
        pos, neg, uni, lin, allDiff = self.collect_data( max_tuples_per_eq, implicid_data=implicid_data, subsampl = subsampl)
        if not pos.is_empty():
            pos.add_to_csp(csp)
        if not neg.is_empty():
            neg.add_to_csp(csp)
        for data in uni:
            data.add_to_csp(csp)
        for data in lin:
            data.add_to_csp(csp)
        for data in allDiff:
            data.add_to_csp(csp)
        return csp
        
    
    def __test__(self, verbose = True, max_num_tuples = 1000000, max_tuples_per_eq = 32):
        """
        unit test to runn through parsing for a file
        """
        self.get_var_domain()
        self.get_constraints(self.tree.getroot().find('constraints'))
        if self.instantiation != []:
            self.preprocessing()
        pos, neg, uni, lin, allDiff = self.collect_data( max_tuples_per_eq, True, max_num_tuples = max_num_tuples)
        csp = self.to_CSP_data()
        return pos, neg, uni, lin, allDiff, csp
    
"""
-------------------------------------------------------------------------------
Classes to represent different kinds of constraints and store associated data
-------------------------------------------------------------------------------
"""

class Constraint(object):
    """
    (abstract) class for Constraints
    """
    
    def __init__(self, variables):
        self.variables = variables   
   
        
class Extension(Constraint):
    """
    class for extension type constraints 
    stores data of an extension and implements functions to export data
    """
    
    def __init__(self, variables, supports, conflicts, negate):
        """
        Constructor for Extension objects

        Parameters
        ----------
        variables :
            list of variables, can contain replacment syntax
        supports : 
            list of tuples the extension allows
        conflicts : 
            list of tuples the extension disallows, either supports or conflicts is empty
        negate : 
            boolean, indicated if the extension specifies allowed or disallowed tuples
        """
        super().__init__(variables)
        self.supports = supports
        self.conflicts = conflicts
        self.negate = negate
    
    
    def preprocessing(self, parser):
        """
        deletes tuples that contain values that are not interesting after instantiation 

        Parameters
        ----------
        parser : 
            xparser object containing the relevant infor for the instantiation
        """
        if parser.instantiation == []: 
            return
        for var in parser.instantiation:
            if var in self.variables:
                i = self.variables.index(var)
                self.variables.remove(var)
                self.supports = [x[:i] + x[i + 1:] for x in self.supports]
                self.conflicts = [x[:i] + x[i + 1:] for x in self.conflicts]
        
        
    def to_tuple(self, parser, variables = []):
        """
        converts data of the extension to Parser_constraint_data containing explicit tuple data

        Parameters
        ----------
        parser : 
            xparser object of the instance
        variables : list, optional
            overwrite variables for the parsing of the extension. The default is [].

        Returns
        -------
        Parser_constraint_data object
        """
        if not variables:
            variables = self.variables
        variables = [parser.var_to_num[var] for var in variables]
        tuple_list = self.conflicts if self.negate else self.supports
        name = 'neg' if self.negate else 'pos'
        if parser.instantiation != []:
            tuple_list = list(filter(lambda x : parser.tuple_in_domain(variables, x), tuple_list))
            if tuple_list == []:
                return Parser_constraint_data(name, self.negate, 1, 0, np.zeros(0, dtype=np.int32), np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.int32))
        tuple_count = len(tuple_list)
        const_idx = np.repeat(0, tuple_count)
        var_idx = np.tile(variables, tuple_count)
        tuple_idx = np.repeat([x for x in range(tuple_count)], len(variables))
        val_idx = np.hstack([[parser.domain[variables[i]][ tup[i]] for i in range(len(tup))] for tup in tuple_list])
        return Parser_constraint_data(name, self.negate, 1, tuple_count, const_idx, tuple_idx, var_idx, val_idx)   

            
    
class Intension(Constraint):
    """
    class for Intension type constraints 
    stores data of an Intension
    """
    
    def __init__(self, expression, args = []):
        """
        Constructor for Intension objects

        Parameters
        ----------
        expression : 
            expression string 
        args : list, optional
            arguments to be fit into replacment syntax. The default is [].
        """
        super().__init__([])
        self.expression = expression
        self.args = args
        
        
    def preprocessing(self, parser):
        """
        replaces variables specified in the intension with their respective set values

        Parameters
        ----------
        parser : 
            xparser object of the instance
        """
        if parser.instantiation == []: 
            return
        for var in parser.instantiation:
            if var in self.args:
                self.args = list(map(lambda x: x.replace(var, str(list(parser.domain[parser.var_to_num[var]].keys())[0])), self.args))                
                
 
    def to_linear(self, parser):
        """
        returns linearzation of the arbitary formula contained in the expression if possible
        main code for linearization contained in intension.py

        Parameters
        ----------
        parser : 
            xpareser object

        Returns
        -------
        Parser_linear_data containing all linear equations needed to model the expression
        """
        var_idx , coeffs, b, comp = intension_to_linear(self.args, self.expression, parser)
        name = str(len(var_idx[0]))
        return Parser_linear_data(name, var_idx, coeffs, b, comp)        
 
       
        
class Comparison(Constraint):
    """
    class for allDifferent type constraints 
    stores data and implements functions to export data
    """
    
    def __init__(self, variables, except_val, diff = True):
        """
        Constructor for Comparison objects

        Parameters
        ----------
        variables : 
            list of variables
        except_val : 
            list of values that are to be excluded from the constraints
        diff : boolean, optional
            unused parameter meant to indicate type of comparison. The default is True.
        """
        super().__init__(variables)
        self.diff = diff
        self.except_val = except_val
        
    
    def preprocessing(self, parser):
        """
        deletes variables contained in instantiation while reducing the domains
        of all other vairables to represent the used value

        Parameters
        ----------
        parser : 
            xparser object
        """
        if parser.instantiation == []: 
            return
        for var in parser.instantiation:
            if var in self.variables:
                val = list(parser.domain[parser.var_to_num[var]].keys())[0]
                self.variables.remove(var)
                for x in self.variables:
                    dom = copy.deepcopy(parser.domain[parser.var_to_num[x]])
                    if val in dom.keys():
                        dom.pop(val)
                        for value, index in dom.items():
                            if value > val:
                                dom[value] = index - 1
                    parser.domain[parser.var_to_num[x]] = dom 
           
        
    def to_tuple(self, parser, variables = [], checkDomains = True, max_num_tuples = 1000000):
        """
        translates allDifferent constraint to list of tuples of arity 2

        Parameters
        ----------
        parser : 
            xparser object
        variables : list, optional
            list of override variables. The default is [].
        checkDomains : boolean, optional
            indicated if domain equivalence has to be checked in parsing. The default is True.
        max_num_tuples : Integer, optional
            number of tuples to stop parsing. The default is 1000000.

        Raises
        ------
        ValueError
            stoped parsing after max_num_tuples tuples

        Returns
        -------
        Parser_constraint_data
            contains explicit tuple data
        """
        if not variables:
            variables = self.variables
        variables = [parser.var_to_num[var] for var in variables]
        if checkDomains:
            for var in variables:
                if parser.domain[var] != parser.domain[variables[0]]:
                    return self.different_domains_to_tuple(parser, variables)
        values = list(filter(lambda val: val not in [parser.domain[variables[0]][x] for x in self.except_val], list(parser.domain[variables[0]].values())))
        tuple_count = len(values)
        num_const = int((len(variables) * (len(variables) - 1)) / 2)
        if tuple_count * num_const > max_num_tuples:
            raise ValueError("To many tuples!")
        val_idx = np.tile(np.repeat(values, 2), num_const)
        var_idx = np.hstack([np.tile([x,y], tuple_count) for x in variables for y in variables if x < y])
        const_idx = np.repeat([x for x in range(num_const)], tuple_count)
        tuple_idx = np.repeat([x for x in range(tuple_count * num_const)], 2)
        name = 'neg'
        return Parser_constraint_data(name, True, num_const, tuple_count * num_const, const_idx, tuple_idx, var_idx, val_idx)


    def different_domains_to_tuple(self, parser, variables):
        """
        translates allDifferent constraint to list of tuples of arity 2
        while taking different Domain into account 
        results in slow processing

        Parameters
        ----------
        parser : 
            xparser object
        variables : 
            list of override varialbes for processing

        Returns
        -------
        Parser_constraint_data object with explicit tuple data
        """
        const_idx = []
        var_idx = []
        val_idx = []
        tuple_idx = []
        num_const = 0
        num_tuple = 0
       
        for var_1 in variables: 
            for var_2 in variables:
                if var_1 <= var_2: 
                    break
                
                length = 0
                for key, value_1 in parser.domain[var_1].items():
                    if key not in self.except_val and key in parser.domain[var_2].keys():
                        val_idx.append([value_1, parser.domain[var_2][key]])
                        length +=1
                if length !=0:
                    const_idx.append(np.repeat(num_const, length))
                    var_idx.append(np.tile([var_1,var_2], length))
                    tuple_idx.append(np.repeat([x + num_tuple for x in range(length)], 2))
                    num_tuple += length
                    num_const += 1
        return Parser_constraint_data('neg', True, num_const, num_tuple, np.hstack(const_idx), np.hstack(tuple_idx), np.hstack(var_idx), np.hstack(val_idx))


    def implicit_data(self, parser, variables=[]):
        """
        export implicit data

        Parameters
        ----------
        parser : 
            xparser object
        variables : list, optional
            override variables used in group parsing. The default is [].

        Raises
        ------
        Exception
            contains except values

        Returns
        -------
        Parser_allDifferent_data object
        """
        if self.except_val != []:
            raise Exception("structure can't be represented as implicit data")
        if not variables:
            variables = self.variables
        var_idx = [[parser.var_to_num[var] for var in variables]]
        name = "allDiff_" + str(len(variables))
        return Parser_allDifferent_data(name, var_idx, parser.num_samples)
        

class LinearEquation(Constraint):
    """
    class for sum type constraints 
    """
    
    def __init__(self, variables, coeffs, operator, operand):
        """
        constructor for LinearEquation

        Parameters
        ----------
        variables : 
            list of variables
        coeffs : 
            list of coefficients
        operator : 
            comparison operator
        operand : 
            operand of the equation
        """
        super().__init__(variables)
        self.coeffs = coeffs
        self.operator = operator
        self.operand = operand
    
    
    def preprocessing(self, parser):
        """
        remove variables that are instantiated and move values to operand

        Parameters
        ----------
        parser : 
            xparser object
        """
        if parser.instantiation == []: 
            return
        for var in parser.instantiation:
            if var in self.variables:
                index = self.variables.index(var)
                self.variables = self.variables[:index] + self.variables[index + 1:]
                self.operand -= self.coeffs[index] * list(parser.domain[parser.var_to_num[var]].keys())[0]
                self.coeffs = self.coeffs[:index] + self.coeffs[index + 1:]
    
    
    def to_tuple(self, parser, variables = [], coeffs = [], checkDomains = False, max_num_tuples = 10000000):
        """
        export to explicit tuple data

        Parameters
        ----------
        parser : 
            xparser object
        variables : list, optional
            overrride variables. The default is [].
        coeffs : list, optional
            coefficients to override. The default is [].
        checkDomains : boolean, optional
            indicated if domain equivalence has to be checked in parsing. The default is False.
        max_num_tuples : integer, optional
            number of tuples to stop parsing. The default is 10000000.

        Raises
        ------
        Exception
            different domains, only raised it checkDomains is True

        Returns
        -------
        Parser_constraint_data
        """
        if not variables:
            variables = self.variables
        variables = [parser.var_to_num[var] for var in variables]
        if len(coeffs) == 0:
            if len(self.coeffs) > 0:
                coeffs = np.array(self.coeffs)
            else:
                coeffs = np.ones(len(variables), dtype = np.int32)
        if checkDomains:
            for var in variables:
                if parser.domain[var] != parser.domain[variables[0]]:
                    raise Exception("linear Equation with different domains")
        domain = list(parser.domain[variables[0]].keys())
        
        tuples, negate = tuple_generator.tuples_from_linear(coeffs, np.array(domain), self.operand, self.operator, max_num_tuples = max_num_tuples)
        
        var_idx = np.tile(variables, len(tuples))
        val_idx = np.reshape(tuples, (len(var_idx))) 
        const_idx = np.repeat(0, len(tuples))
        tuple_idx = np.repeat([x for x in range(len(tuples))], len(variables))
        name = 'neg' if negate else 'pos'
        return Parser_constraint_data(name, negate, 1, len(tuples), const_idx, tuple_idx, var_idx, val_idx)
    
    
    def implicit_data(self, parser, variables = [], coeffs = [], checkDomains = True, max_num_tuples = 32):
        """
        export implicit data

        Parameters
        ----------
        parser : 
            xparser object
        variables : list, optional
            override variables. The default is [].
        coeffs : list, optional
            override coefficients. The default is [].
        checkDomains : boolean, optional
            indicated if domain equivalence has to be checked in parsing. The default is True.
        max_num_tuples : integer, optional
            unused number of tuples to stop parsing. The default is 32.

        Returns
        -------
        Parser_linear_data
        """
        if not variables:
            variables = self.variables
        variables = [parser.var_to_num[var] for var in variables]
        arity = len(variables)
        name = str(arity)
        if len(coeffs) ==0:
            if len(self.coeffs) >0:
                coeffs = self.coeffs
            else:
                coeffs = np.ones(len(variables), dtype= np.int32)
        var_idx = np.array([variables])
        b = self.operand
        return Parser_linear_data(name, var_idx, coeffs, [b], [self.operator])


        
class Group(object):
    """
    class for groups of constraints
    """
    
    def __init__(self, constraint, arguments):
        """
        constructor for groups

        Parameters
        ----------
        constraint : 
            generic constraint object
        arguments : 
            list of argumetns to be applied to constraint
        """
        self.const = constraint
        self.arguments = arguments
        
    
    def preprocessing(self, parser):
        """
        call preprocessing of constraint and modify arguemnts

        Parameters
        ----------
        parser : 
            xparser object
        """
        if parser.instantiation == []: 
            return
        new_args = []
        for arg in self.arguments:
            for var in parser.instantiation:
                split = False
                if var in arg:
                    const = copy.deepcopy(self.const)
                    const.variables = XParser.replace_var(const.variables, arg)
                    const.preprocessing(parser)
                    parser.constraints.append(const)
                    split = True
                    continue
            if not split:
                new_args.append(arg)
        self.arguments = new_args
    
    
    def get_data(self, parser, implicit_data, max_num_tuples, subsampl = True):
        """
        calls right parsing methods based on parameters

        Parameters
        ----------
        parser : 
            xparser object
        implicit_data : 
            boolean to indicate if only explicit data is to be parsed
        max_num_tuples : 
            number of tuples to stop parsing at
        subsampl : boolean, optional
            indicate if subsampling is done, unused here. The default is True.

        Returns
        -------
        Parser_data
        """
        if implicit_data and type(self.const) == LinearEquation:
            return self.linear_implicit_data(parser,max_num_tuples)
        elif implicit_data and type(self.const) == Comparison:
            return self.allDifferent_implicit_data(parser)
        else:
            return self.to_tuple(parser, subsampl)
    
    
    def to_tuple(self, parser, subsampl = True):
        """
        calls tuple generating methods for group

        Parameters
        ----------
        parser : 
            xparser object
        subsampl : boolean, optional
            indicate if subsampling is done, unused here. The default is True.

        Raises
        ------
        Exception
            unimplemented type of group

        Returns
        -------
        Parser_data
        """
        if type(self.const) == LinearEquation:
            return self.linear_group_to_tuple(parser, subsampl)
        elif type(self.const) == Extension:
            return self.extension_group_to_tuple(parser, subsampl)
        elif type(self.const) == Comparison:
            return self.allDifferent_group_to_tuple(parser, subsampl)
        else:
            raise Exception()
    
    
    def extension_group_to_tuple(self, parser, subsampl = True):
        """
        parse a group of extension to explicit data,
        implements fast parsing for extesnion only files

        Parameters
        ----------
        parser : 
            xparser object
        subsampl : boolean, optional
            indicate if subsampling is done, unused here. The default is True.

        Returns
        -------
        Parser_data
        """
        if parser.plain_extension():
            name = 'neg' if self.const.negate else 'pos'
            tuple_list = self.const.conflicts if self.const.negate else self.const.supports
            tuple_count = len(tuple_list)
            if subsampl and tuple_count >= parser.num_samples:
                arity = len(tuple_list[0])
                var_idx = np.stack([[[parser.var_to_num[v] for v in args] for _ in range(tuple_count)] for args in self.arguments], axis = 0)
                arr = np.array([[parser.domain[0][tup[i]] for i in range(len(tup))] for tup in tuple_list])
                val_idx = np.repeat(arr[None,...] , len(self.arguments), axis = 0)
                name = name + "_" + str(tuple_count) + "_" + str(arity)
                return Parser_uniform_data(name, self.const.negate, parser.num_samples, var_idx, val_idx)
            else:
                var_idx = np.hstack([np.tile([parser.var_to_num[v] for v in args], tuple_count) for args in self.arguments])
                val_idx = np.tile(np.hstack([[parser.domain[0][tup[i]] for i in range(len(tup))] for tup in tuple_list]), len(self.arguments))
                const_idx = np.repeat(list(range(len(self.arguments))), tuple_count)
                tuple_idx = np.repeat(list(range(len(self.arguments) * tuple_count)), len(tuple_list[0]))
                return Parser_constraint_data(name, self.const.negate, len(self.arguments), len(self.arguments) * tuple_count, const_idx, tuple_idx, var_idx, val_idx)
        
        ext_list = []
        for args in self.arguments:
            variables = XParser.replace_var(self.const.variables, args)
            ext_list.append(self.const.to_tuple(parser, variables))
        name = 'neg' if ext_list[0].negate else 'pos'
        if subsampl and ext_list[0].num_tuple >= parser.num_samples:
            #reshape to uniform
            arity = int(len(ext_list[0].tuple_idx) / ext_list[0].num_tuple)
            name = name + "_" + str(ext_list[0].num_tuple) + "_" + str(arity)
            val_idx = np.vstack([XParser.to_3d_array(ext_list[x].num_tuple, ext_list[x].val_idx) for x in range(len(self.arguments))])
            var_idx = np.vstack([XParser.to_3d_array(ext_list[x].num_tuple, ext_list[x].var_idx) for x in range(len(self.arguments))])
            return Parser_uniform_data(name, ext_list[0].negate, parser.num_samples, var_idx, val_idx)
        else:
            var_idx = np.hstack([data.var_idx for data in ext_list])
            val_idx = np.hstack([data.val_idx for data in ext_list])
            const_idx = np.hstack([ext_list[x].const_idx + x for x in range(len(self.arguments))])
            tuple_idx = np.hstack([ext_list[x].tuple_idx + (x * ext_list[x].num_tuple) for x in range(len(self.arguments))])
            return Parser_constraint_data(name, ext_list[0].negate, len(self.arguments), sum([data.num_tuple for data in ext_list]), const_idx, tuple_idx, var_idx, val_idx)
    
    
    def allDifferent_group_to_tuple(self, parser, subsampl = True):
        """
        parse a group of allDifferent constraints to explicit data

        Parameters
        ----------
        parser : 
            xparser object
        subsampl : boolean, optional
            indicate if subsampling is done, unused here. The default is True.

        Returns
        -------
        Parser_data
        """
        one_by_one = False
        for args in self.arguments:
            for arg in args:
                if parser.domain[parser.var_to_num[arg]] != parser.domain[parser.var_to_num[self.arguments[0][0]]]:
                   one_by_one = True
            if len(args) != len(self.arguments[0]):
                one_by_one = True
        if one_by_one:
            ret = Parser_constraint_data('neg', True)
            for args in self.arguments:
                ret.append_data(self.const.to_tuple(parser, XParser.replace_var(self.const.variables, args)))
            return ret
        else:
            data = self.const.to_tuple(parser, XParser.replace_var(self.const.variables, self.arguments[0]))
            ind_tuple_count = int(data.num_tuple / data.num_const)
            var_to_array = lambda args: np.hstack([np.tile([x,y], ind_tuple_count) for x in args for y in args if x < y])
            arg_count = len(self.arguments)
            name = 'neg' if data.negate else 'pos'
            if subsampl and ind_tuple_count >= parser.num_samples:
                #reshape to uniform
                arity = int(len(data.tuple_idx) / data.num_tuple)
                name = name + "_" + str(ind_tuple_count) + "_" + str(arity)
                val_idx = XParser.to_stacked_3d_array(data.num_const, ind_tuple_count, data.val_idx)
                #multiply for arguments
                val_idx = np.tile(val_idx, (arg_count, 1, 1))
                var_3d = lambda args: XParser.to_stacked_3d_array(data.num_const, ind_tuple_count, var_to_array(args))
                var_idx = np.vstack([var_3d(list(map(lambda x: parser.var_to_num[x], args))) for args in self.arguments])
                return Parser_uniform_data(name, data.negate, parser.num_samples, var_idx, val_idx)
            else:
                const_idx = np.hstack([data.const_idx + x * data.num_const for x in range(arg_count)])
                tuple_idx = np.hstack([data.tuple_idx + x * data.num_tuple for x in range(arg_count)])
                num_tuple = data.num_tuple * arg_count
                var_idx = np.hstack([var_to_array(list(map(lambda x: parser.var_to_num[x], args))) for args in self.arguments])
                val_idx = np.tile(data.val_idx, arg_count)
                return Parser_constraint_data(name, data.negate, len(self.arguments) * data.num_const, num_tuple, const_idx, tuple_idx, var_idx, val_idx) 
    
    
    def allDifferent_implicit_data(self, parser):
        """
        call allDifferent to implicit data for all arguments

        Parameters
        ----------
        parser : 
            xparser object
            
        Returns
        -------
        data : 
            merged implicit data as Parser_allDifferent_data
        """
        data = self.const.implicit_data(parser, XParser.replace_var(self.const.variables, self.arguments[0]))
        for args in self.arguments[1:]:
            data.append_data(self.const.implicit_data(parser, XParser.replace_var(self.const.variables, args)))
        return data
    
    
    def linear_group_to_tuple(self, parser, subsampl = True):
        """
        linearEquation group to tuple data
        caution, does not support args for coeffs
        
        Parameters
        ----------
        parser : 
            xparser object
        subsampl : boolean, optional
            unused here. The default is True.

        Raises
        ------
        Exception
            different domains for vairables in linear equations

        Returns
        -------
        Parser_data
        """
        one_by_one = False
        for args in self.arguments:
            if len(args) != len(self.arguments[0]):
                one_by_one = True
            for arg in args:
                if parser.domain[parser.var_to_num[arg]] != parser.domain[parser.var_to_num[self.arguments[0][0]]]:
                   raise Exception('different domains in linear equations not implemented')
        if one_by_one:
            for args in self.arguments:
                parser.constraints.append(LinearEquation(XParser.replace_var(self.const.variables, args), self.const.coeffs, self.const.operator, self.const.operand))
            return Parser_constraint_data("neg", True)
        data = self.const.to_tuple(parser, XParser.replace_var(self.const.variables, self.arguments[0]))
        arg_count = len(self.arguments)
        name = 'neg' if data.negate else 'pos'
        if subsampl and data.num_tuple >= parser.num_samples:
            #reshape to uniform
            arity = int(len(data.tuple_idx) / data.num_tuple / data.num_const)
            name = name + "_" + str(data.num_tuple) + "_" + str(arity)
            val_idx = XParser.to_3d_array(data.num_tuple, data.val_idx)
            #multiply for arguments
            val_idx = np.tile(val_idx, (arg_count, 1, 1))
            var_idx = np.vstack([np.tile([parser.var_to_num[var] for var in args], (1, data.num_tuple, 1)) for args in self.arguments])
            return  Parser_uniform_data(name, data.negate, parser.num_samples, var_idx, val_idx)
        else:
            const_idx = np.hstack([data.const_idx + x for x in range(arg_count)])
            tuple_idx = np.hstack([data.tuple_idx + x * data.num_tuple for x in range(arg_count)])
            num_tuple = data.num_tuple * arg_count
            var_idx = np.hstack([np.tile([parser.var_to_num[var] for var in args], data.num_tuple) for args in self.arguments])
            val_idx = np.tile(data.val_idx, arg_count)
            return Parser_constraint_data(name, data.negate, arg_count * data.num_const, num_tuple, const_idx, tuple_idx, var_idx, val_idx) 
        
    
    def linear_implicit_data(self, parser, max_num_tuples):
        """
        call linearEquation to implicit data for all arguments

        Parameters
        ----------
        parser : 
            xparser object
        max_num_tuples : 
            unused here

        Returns
        -------
        Parser_linear_data
        """
        data = self.const.implicit_data(parser, XParser.replace_var(self.const.variables, self.arguments[0]), XParser.replace_coeff(self.const.coeffs, self.arguments[0]), max_num_tuples)
        for args in self.arguments[1:]:
            data.append_data(self.const.implicit_data(parser, XParser.replace_var(self.const.variables, args), XParser.replace_coeff(self.const.coeffs, args), max_num_tuples))
        return data
    
    
    def split_group(self):
        """
        split groups arrording to different arities

        Returns
        -------
        list of new groups
        """
        groups = []
        for args in self.arguments:
            num = len(args)
            arguments = [args]
            for group_num, group_arguments in groups:
                if group_num == num:
                    arguments = arguments + group_arguments
                    groups.remove((group_num, group_arguments))
            groups.append((num, arguments))
        if len(groups) > 1:
            ret = []
            for num, arguments in groups:
                ret.append(Group(self.const, arguments))
            return ret
        else: 
            return [self]
                    
                                     

class Parser_data(object):
    """
    (abstract) class for data of the parser
    """
    def __init__(self, name, negate, var_idx = np.zeros(0, dtype = np.int32), val_idx = np.zeros(0, dtype = np.int32)):
        """
        constructor

        Parameters
        ----------
        name : 
            name of the class
        negate : 
            boolean indicating if explicit data is negated
        var_idx : numpy array, optional
            variable indices. The default is np.zeros(0, dtype = np.int32).
        val_idx : numpy array, optional
            value indices. The default is np.zeros(0, dtype = np.int32).
        """
        self.name = name
        self.negate = negate
        self.val_idx = val_idx
        self.var_idx = var_idx
  
    
  
class Parser_constraint_data(Parser_data):
    """
    class for explicit tuple data
    """
    
    def __init__(self, name, negate, num_const = 0, num_tuple = 0, const_idx = np.zeros(0, dtype = np.int32),\
                 tuple_idx = np.zeros(0, dtype = np.int32), var_idx = np.zeros(0, dtype = np.int32), val_idx = np.zeros(0, dtype = np.int32)):
        """
        constructor

        Parameters
        ----------
        name : 
            name of the class
        negate : 
            boolean indicating if explicit data is negated
        num_const : TYPE, optional
            number of constrints. The default is 0.
        num_tuple : TYPE, optional
            number of tuples. The default is 0.
        const_idx : TYPE, optional
            constraint indices. The default is np.zeros(0, dtype = np.int32).
        tuple_idx : TYPE, optional
            tuple indices. The default is np.zeros(0, dtype = np.int32).
        var_idx : numpy array, optional
            variable indices. The default is np.zeros(0, dtype = np.int32).
        val_idx : numpy array, optional
            value indices. The default is np.zeros(0, dtype = np.int32).
        """
        super().__init__(name, negate, var_idx, val_idx)
        self.num_const = num_const
        self.num_tuple = num_tuple
        self.const_idx = const_idx
        self.tuple_idx = tuple_idx
    
    
    def is_empty(self):
        """
        check if the object contains data

        Returns
        -------
        boolean
        """
        return False if self.num_const != 0 else True    
    
    
    def to_uniform(self, num_samples):
        """
        transform data to Parser_uniform_data

        Parameters
        ----------
        num_samples : 
            unused

        Returns
        -------
        Parser_uniform_data
        """
        ind_tuple_count = int(self.num_tuple / self.num_const)
        var_idx = XParser.to_stacked_3d_array(self.num_const, ind_tuple_count, self.var_idx)
        val_idx = XParser.to_stacked_3d_array(self.num_const, ind_tuple_count, self.val_idx)
        name = self.name + "_" + str(var_idx.shape[1]) + "_" + str(var_idx.shape[2])
        return Parser_uniform_data(name, self.negate, num_samples, var_idx, val_idx)
    
    
    def append_data(self, data):
        """
        append new data to object

        Parameters
        ----------
        data : 
            another Parser_constraint_data object to merge

        Raises
        ------
        ValueError
            data is of wrong type

        Returns
        -------
        data : 
            given data
        """
        if type(data) != Parser_constraint_data:
            raise ValueError("expected Parser_constraint_data object")
        if self.name != data.name:
            print(self.name, " vs ", data.name)
            raise ValueError("can not merge different types of data")
        self.const_idx = np.hstack([self.const_idx, data.const_idx + self.num_const])
        self.tuple_idx = np.hstack([self.tuple_idx , data.tuple_idx + self.num_tuple])
        self.val_idx = np.hstack([self.val_idx , data.val_idx])
        self.var_idx = np.hstack([self.var_idx , data.var_idx])
        self.num_tuple += data.num_tuple
        self.num_const += data.num_const
    
    
    def form_tensor(self):
        """
        internal data representation changes to torch tensors
        """
        if type(self.var_idx) == list:
            self.concatenate_data()
        self.const_idx = torch.LongTensor(self.const_idx)
        self.tuple_idx = torch.LongTensor(self.tuple_idx)
        self.val_idx = torch.LongTensor(self.val_idx)
        self.var_idx = torch.LongTensor(self.var_idx)
    
    
    def add_to_csp(self, csp):
        """
        add data to csp data object

        Parameters
        ----------
        csp : 
            csp_data object
        """
        if type(self.var_idx) != torch.Tensor:
            self.form_tensor()
        csp.add_constraint_data(self.negate, self.const_idx, self.tuple_idx, self.var_idx, self.val_idx)
        
        
        
class Parser_uniform_data(Parser_data):
    """
    class for explicit tuple data as uniform representation
    """
    
    def __init__(self, name, negate, num_samples, var_idx = np.zeros(shape = (1, 1, 1)), val_idx = np.zeros(shape = (1, 1, 1))):
        """
        constructor

        Parameters
        ----------
        name : 
            data name to identify
        negate : 
            indicate data negation
        num_samples : 
            unused
        var_idx : TYPE, optional
            variable indices. The default is np.zeros(shape = (1, 1, 1)).
        val_idx : TYPE, optional
            value indices. The default is np.zeros(shape = (1, 1, 1)).
        """
        super().__init__(name, negate, var_idx, val_idx)
        self.num_samples = num_samples
    
    
    def is_empty(self):
        """
        check if the object contains data

        Returns
        -------
        boolean
        """
        return False if self.var_idx.shape != (1, 1, 1) else True
    
    
    def form_tensor(self):
        """
        internal data representation changes to torch tensors
        """
        self.var_idx = torch.LongTensor(self.var_idx)
        self.val_idx = torch.LongTensor(self.val_idx)

        
    def append_data(self, data):
        """
        append new data to object

        Parameters
        ----------
        data : 
            another Parser_uniform_data object to merge

        Raises
        ------
        ValueError
            data is of wrong type
        """
        if type(data) != Parser_uniform_data:
            raise ValueError("expected Parser_uniform_data object")
        if self.name != data.name:
            print(self.name, " vs ", data.name)
            raise ValueError("can not merge different types of data")
        self.var_idx = np.vstack([self.var_idx, data.var_idx])
        self.val_idx = np.vstack([self.val_idx, data.val_idx])

        
    def add_to_csp(self, csp):
        """
        add data to csp data object

        Parameters
        ----------
        csp : 
            csp_data object
        """
        if type(self.var_idx) != torch.Tensor:
            self.form_tensor()
        csp.add_uniform_constraint_data(self.negate, self.var_idx, self.val_idx)



class Parser_linear_data(Parser_data):
    """
    class for implicit linear equation data
    """
    
    def __init__(self, name, var_idx, coeffs, b, comp, eq_reduction = True):
        """
        constructor

        Parameters
        ----------
        name : 
            identifier string used in merge
        var_idx : 
            vaiable indices
        coeffs : 
            list of coefficients
        b : 
            operand list of equations
        comp : 
            operator list
        eq_reduction : boolean, optional
            indicate if eqivalence equations are reduced to 2 le equations. The default is True.
        """
        super().__init__(name, True, [var_idx])
        self.coeffs = [coeffs]
        self.b = b
        self.comp = comp
        self.operator_reduction(eq_reduction)
    
    
    def append_data(self, data):
        """
        append new data to object

        Parameters
        ----------
        data : 
            another Parser_linear_data object to merge

        Raises
        ------
        ValueError
            data is of wrong type
        """
        if type(data) != Parser_linear_data:
            raise ValueError("expected Parser_linear_data object")
        if self.name != data.name:
            print(self.name, " vs ", data.name)
            raise ValueError("can not merge different types of data")
        self.coeffs += data.coeffs
        self.var_idx += data.var_idx
        self.comp += data.comp
        self.b += data.b
        
        
    def operator_reduction(self, eq_reduction):
        """
        reduce linear equations to only le and ne containing the same requirments
        
        warning: if multiple Parser_linear_data objects have been merged 
                 the effect of this function is undefined
        
        Parameters
        ----------
        eq_reduction : 
            boolean to indicate if reduction is to be aplied to eq
        """
        for i in range(len(self.comp)):
            if eq_reduction and self.comp[i] == 'eq': 
                self.comp[i] = 'le'
                self.coeffs.append([-x for x in self.coeffs[i]])
                self.var_idx.append(self.var_idx[i])
                self.comp.append('le')
                self.b.append(-self.b[i])
            elif self.comp[i] == 'gt':
                self.comp[i] = 'le'
                self.coeffs[i] = [-x for x in self.coeffs[i]]
                self.b[i] = self.b[i] - 1
            elif self.comp[i] == 'lt':
                self.comp[i] = 'le'
                self.b[i] -= 1
            elif self.comp[i] == 'ge':
                self.comp[i] = 'le'
                self.coeffs[i] = [-x for x in self.coeffs[i]]
                self.b[i] = -self.b[i]
    
    
    def evaluate_trivial(self):
        """
        evaluates equations without variables

        Raises
        ------
        Exception
            instance is unsatisfiable
        """
        operations = {'gt': op.gt,
                         'lt': op.lt,
                         'ge': op.ge,
                         'le': op.le,
                         'eq': op.eq,
                         'ne': op.ne}
        if self.name == "0":
            for i in range(len(self.comp)):
                if not operations[self.comp[i]](0, self.b[i]):
                    raise Exception('instance is unsatisfialbe after preprocessing')
    
    
    def form_tensor(self):
        """
        internal data representation changes to torch tensors
        """
        self.var_idx = torch.LongTensor(np.vstack(self.var_idx))
        self.coeffs = torch.LongTensor(np.vstack(self.coeffs))


    def add_to_csp(self, csp):
        """
        add data to csp data object

        Parameters
        ----------
        csp : 
            csp_data object
        """
        if self.name == "0":
            self.evaluate_trivial()
            return
        if type(self.var_idx) != torch.Tensor:
            self.form_tensor()
        csp.add_linear_constraint_data(self.var_idx, self.coeffs, torch.tensor(self.b), self.comp)



class Parser_allDifferent_data(Parser_data):
    """
    class for implicit allDifferent data
    """
    def __init__(self, name, var_idx, num_samples):
        """
        constructor

        Parameters
        ----------
        name : 
            identifier used in merging
        var_idx : 
            variable indices
        num_samples : 
            unused
        """
        super().__init__(name, True, var_idx)
        self.num_samples = num_samples
        
        
    def append_data(self, data):
        """
        append new data to object

        Parameters
        ----------
        data : 
            another Parser_allDifferent_data object to merge

        Raises
        ------
        ValueError
            data is of wrong type
        """
        if type(data) != Parser_allDifferent_data:
            raise ValueError("expected Parser_allDifferent_data object")
        if self.name != data.name:
            print(self.name, " vs ", data.name)
            raise ValueError("can not merge different types of data")
        self.var_idx = np.vstack([self.var_idx, data.var_idx])
        
        
    def form_tensor(self):
        """
        internal data representation changes to torch tensors
        """
        self.var_idx = torch.LongTensor(self.var_idx)
    
    
    def add_to_csp(self, csp):
        """
        add data to csp data object

        Parameters
        ----------
        csp : 
            csp_data object
        """
        if type(self.var_idx) != torch.Tensor:
            self.form_tensor()
        csp.add_all_different_constraint_data(self.var_idx)


if __name__ == '__main__':
    """
    unit test execution
    """
    test = XParser("../../data/tests/unit_test_01.xml")
    test.__test__(max_tuples_per_eq = 3, max_num_tuples = 10000000000)
    test = XParser("../../data/tests/unit_test_02.xml")
    test.__test__(max_tuples_per_eq = 3, max_num_tuples = 10000000000)
    test = XParser("../../data/tests/unit_test_03.xml")
    test.__test__(max_tuples_per_eq = 3, max_num_tuples = 10000000000)
    test = XParser(".\\data\\tests\\unit_test_04.xml")
    test.__test__(max_tuples_per_eq = 3, max_num_tuples = 10000000000)