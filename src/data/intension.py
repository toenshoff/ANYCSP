from enum import Enum
import copy
import re

relations = ['gt','le','ge','lt','eq','ne']
inverse_relation = {'gt':'le','le':'gt','ge':'lt','lt':'ge','eq':'ne','ne':'eq'}
red_logic = ['and','xor','iff','imp']

class Node_type(Enum):
    """
    enum of all types of nodes used to represent intensions
    """
    VAL = 1
    VAR = 2
    OP = 3
    ARG = 4

class Node():
    """
    main class for tree representation of intension formula
    """

    def __init__(self, children = None, value = None, node_type = None):
        """
        constructor of a new node

        Parameters
        ----------
        children : list, optional
            children of the node. The default is None.
        value : optional
            value contained in the node, type depends on node_type. The default is None.
        node_type : Node_type, optional
            type of the node. The default is None.
        """
        self.children = children
        self.value = value
        self.node_type = node_type
        

    def from_string_rec(self, string):
        """
        reades an intension from a given string and returns the root of a tree
        representing the expression
        
        Parameters
        ----------
        string :
            text represnetation of the intension
        """
        if match := re.match(r"^(?P<operator>\w+)\((?P<rest>.*)", string):
            self.value = match.group('operator')
            self.node_type = Node_type.OP
            rest = match.group('rest')
            self.children = []
            while True:
                rest = re.sub("^,","", rest)
                child, rest = Node().from_string_rec(rest)
                self.children.append(child)
                if not re.match(r"^,", rest):
                    break
            return self, re.sub("^\)", "", rest)
                
        elif match := re.match(r"^(?P<arg>\%\d+)(?P<rest>[),].*)", string):
            self.value = match.group('arg')
            self.node_type = Node_type.ARG
            return self, match.group('rest')
            
        elif match := re.match(r"^(?P<val>[-0-9]*)(?P<rest>[),].*)", string):
            self.value = int(match.group('val'))
            self.node_type = Node_type.VAL
            return self, match.group('rest')
        
        elif match := re.match(r"^(?P<var>[\w\][]*)(?P<rest>[),].*)", string):
            self.value = match.group('var')
            self.node_type = Node_type.VAR
            return self, match.group('rest')
        else:
            raise Exception(string, "could not be evaluated as intension")


    def preorder_print(self):
        """
        method to create string representation for the intension from self as root node

        Returns
        -------
        string : 
            preorder notation formula
        """
        string = str(self.value)
        if self.node_type == Node_type.OP:
            string += "("
            for child in self.children:
                string += child.preorder_print() + ","
            string = string[:-1]
            string += ")"
        return string
        
    
    def reduce_logic(self, isRoot):
        """
        applies arithmetisation to formula tree to represent arbitary formula as linear formula
        
        reduction has certain formal requirments: 
            - intension does not use aritmetic operations on boolean values
            - currently no eq with more than 2 children in the root
            - currently no not() on bools 0 or 1

        Parameters
        ----------
        isRoot : 
            parameter indicates if given node is root node of the equation
        """
        if self.node_type == Node_type.OP:
            og_children = self.children
            if self.value == 'and':
                neg = []
                for child in og_children:
                    neg.append(Node(children= [child], value = 'not', node_type = Node_type.OP))
                orNode = Node(children = neg, value = 'or', node_type = Node_type.OP) 
                self.value = 'not'
                self.children = [orNode]
            elif self.value == 'imp':
                neg = Node(children = [og_children[0]], value = 'not', node_type = Node_type.OP)
                self.value = 'or'
                self.children = [neg, og_children[1]]
            elif self.value == 'iff':
                #could be reduced to not(xor())
                self.value = 'and'
                children= []
                for child in og_children[1:]:
                    neg_x = Node(children = [og_children[0]], value = 'not', node_type = Node_type.OP)
                    neg_y = Node(children = [child], value = 'not', node_type = Node_type.OP)
                    orNode_neg = Node(children = [neg_x, neg_y], value = 'or', node_type = Node_type.OP)
                    orNode = Node(children = [og_children[0], child], value = 'or', node_type = Node_type.OP)
                    neg_1 = Node(children = [orNode_neg], value = 'not', node_type = Node_type.OP)
                    neg_2 = Node(children = [orNode], value = 'not', node_type = Node_type.OP)
                    children.append(Node(children = [neg_1, neg_2], value = 'or', node_type = Node_type.OP))
                if len(og_children) == 2:
                    self.value = 'or'
                    self.children= children[0].children
                else:
                    self.children = children
                    self.reduce_logic(isRoot)
            elif self.value == 'xor':
                neg_1 = Node(children = [og_children[0]], value = 'not', node_type = Node_type.OP)
                neg_2 = Node(children = [og_children[1]], value = 'not', node_type = Node_type.OP)
                or_1 = Node(children = [neg_1, og_children[1]], value = 'or', node_type = Node_type.OP)
                or_2 = Node(children = [neg_2, og_children[0]], value = 'or', node_type = Node_type.OP)
                negOr_1 = Node(children = [or_1], value = 'not', node_type = Node_type.OP)
                negOr_2 = Node(children = [or_2], value = 'not', node_type = Node_type.OP)
                if len(og_children) > 2:
                    self.children = [Node(children = [negOr_1, negOr_2], value = 'or', node_type = Node_type.OP)] + og_children[2:]
                    self.reduce_logic(isRoot)
                else:
                    self.value = 'or'
                    self.children = [negOr_1, negOr_2]
            elif self.value == 'lt' and not isRoot:
                dist = Node(children = [og_children[0], og_children[1]], value = 'dist', node_type = Node_type.OP)
                add = Node(children = [dist, og_children[1]], value = 'add', node_type = Node_type.OP)
                subb = Node(children = [add, og_children[0]], value = 'sub', node_type = Node_type.OP)
                scalar = Node(value = 0, node_type = Node_type.VAL)
                self.value = 'gt'
                self.children = [subb, scalar]
            elif self.value == 'le' and not isRoot:
                dist = Node(children = [og_children[0], og_children[1]], value = 'dist', node_type = Node_type.OP)
                add = Node(children = [dist, og_children[1]], value = 'add', node_type = Node_type.OP)
                subb = Node(children = [add, og_children[0]], value = 'sub', node_type = Node_type.OP)
                scalar = Node(value = -1, node_type = Node_type.VAL)
                self.value = 'gt'
                self.children = [subb, scalar]
            elif self.value == 'gt' and not isRoot:
                dist = Node(children = [og_children[0], og_children[1]], value = 'dist', node_type = Node_type.OP)
                add = Node(children = [dist, og_children[0]], value = 'add', node_type = Node_type.OP)
                subb = Node(children = [add, og_children[1]], value = 'sub', node_type = Node_type.OP)
                scalar = Node(value = 0, node_type = Node_type.VAL)
                self.value = 'gt'
                self.children = [subb, scalar]
            elif self.value == 'ge' and not isRoot:
                dist = Node(children = [og_children[0], og_children[1]], value = 'dist', node_type = Node_type.OP)
                add = Node(children = [dist, og_children[0]], value = 'add', node_type = Node_type.OP)
                subb = Node(children = [add, og_children[1]], value = 'sub', node_type = Node_type.OP)
                scalar = Node(value = -1, node_type = Node_type.VAL)
                self.value = 'gt'
                self.children = [subb, scalar]
            elif self.value == 'ne' and not isRoot:
                dist = Node(children = [og_children[0], og_children[1]], value = 'dist', node_type = Node_type.OP)
                scalar = Node(value = 0, node_type = Node_type.VAL)
                self.value = 'gt'
                self.children = [dist, scalar]
            elif self.value == 'eq' and not isRoot:
                # 2* le + and 
                self.value = 'and'
                children= []
                for child in og_children[1:]:
                    le1 = Node(children = [og_children[0], child], value = 'le', node_type = Node_type.OP)
                    le2 = Node(children = [child, og_children[0]], value = 'le', node_type = Node_type.OP)
                    children.append(le1)
                    children.append(le2)
                self.children = children
                self.reduce_logic(isRoot)
                
        if self.children:
            for child in self.children:
                child.reduce_logic(False)

    
    @staticmethod
    def invert_dict(dict):
        """
        inverts all coeffs (w.r.t. (Z,+))
        thus represents subtracting the contents of the dict from the linear equation


        Parameters
        ----------
        dict : 
            dictionary to be inverted

        Returns
        -------
        inverted dictionary
        """
        tmp = copy.deepcopy(dict)
        for i in dict.keys():
            if i == 'comp':
                continue
            tmp[i] *= -1
        return tmp


    @staticmethod
    def merge_dict(dict_1, dict_2):
        """
        adds up coeffs of a linear equation

        Parameters
        ----------
        dict_1 : 
            dictionray to be merged
        dict_2 : 
            dictionray to be merged

        Returns
        -------
        merged dictionary
        """
        tmp = copy.deepcopy(dict_1)
        for key in dict_2.keys():
            if key == 'comp':
                continue
            if key in dict_1.keys():
                tmp[key] += dict_2[key]
            else:
                tmp[key] = dict_2[key]
        return tmp                    
    
    
    @staticmethod
    def get_arrays_from_dict(dict, parser):
        """
        extract the data from dictiory to pass it back to the parser

        Parameters
        ----------
        dict : 
            dictionary representing an equation
        parser : 
            xparser object

        Returns
        -------
        list
            variables
        list
            coefficients
        b : 
            operator
        comp : 
            operand
        """
        if type(dict.get('s')) == int:
            b = [-dict['s']]
            del dict['s']
        else:
            b = 0
        if dict.get('comp'):
            comp = dict['comp']
            del dict['comp']
        else:
            raise Exception()
        return [parser.var_to_num[i] for i in dict.keys()], list(dict.values()), b, comp


    def normalize(self, isRoot):
        """
        main method to get dictionaries representing linear equations form parsing tree
        
        warning: does not work for 3 or more elements to be compared at top level

        Parameters
        ----------
        isRoot : 
            indicated if current node is root node of the formula  

        Returns
        -------
        list of dicitionaries of linear equations
        """
        if self.node_type == Node_type.VAR:
            return [{self.value: 1}]
        elif self.node_type == Node_type.VAL:
            return [{'s': int(self.value)}]
        elif self.node_type == Node_type.ARG:
            raise Exception()
            #return [{self.value: 1}]
        else:
            left = self.children[0].normalize(False)
            if isRoot and self.value in relations:
                #TODO eq für k>=3
                right = self.children[1].normalize(False)
                res = [Node.merge_dict(i, Node.invert_dict(j)) for i in left for j in right]
                for dict in res:
                    if 'comp' in dict.keys():
                        raise ValueError
                    else:
                        dict['comp'] = self.value
                return res
            elif self.value in relations:
                if self.value == 'gt':
                    right = self.children[1].normalize(False)
                    res = [Node.merge_dict(i, Node.invert_dict(j)) for i in left for j in right]
                    for dict in res:
                        if 'comp' in dict.keys():
                            raise ValueError
                        else:
                            dict['comp'] = self.value
                    return res
                else:
                    raise Exception()
                
            elif self.value == 'or':
                parts = [child.normalize(False) for child in self.children]
                res = parts[0]
                for part in parts[1:]:
                    for dict in res+ part:
                        if 'comp' in dict.keys():
                            if dict['comp'] == 'gt':
                                continue
                            else:
                                raise Exception()
                        else:
                            raise Exception()
                    res = [Node.merge_dict(r, dict) for r in res for dict in part]
                return res
            elif self.value == 'not':
                if 'comp' in left[0].keys():
                    return [Node.invert_dict(i) for i in left]
                else:
                    print("warning: not() on dict without 'comp'")
                    return [Node.invert_dict(i) for i in left]
            if self.value == 'neg':
                return [Node.invert_dict(i) for i in left]
            elif self.value == 'add':
                right = self.children[1].normalize(False)
                return [Node.merge_dict(i,j) for i in left for j in right]
            elif self.value == 'sub':
                right = [Node.invert_dict(i) for i in self.children[1].normalize(False)] 
                return [Node.merge_dict(i,j) for i in left for j in right]
            elif self.value == 'abs':
                res = left + [Node.invert_dict(i) for i in left]
                return res
            elif self.value == 'dist':
                right = [Node.invert_dict(i) for i in self.children[1].normalize(False)]
                merged = [Node.merge_dict(i,j) for i in left for j in right]
                return merged + [Node.invert_dict(i) for i in merged]
            else:
                print('Operator not implemented')
                exit()
                                  

def intension_to_linear(params, expression, parser):
    """
    method to parse an intension

    Parameters
    ----------
    params : 
        arguments to be inserted into replacment nodes
    expression : 
        string representing the expresiion in preorder notation
    parser : 
        xparser object

    Returns
    -------
    var_idx : 
        variable inidces
    coeffs : 
        list of coefficients
    b : 
        scalar operand
    comp : 
        operators
    """    
    for index, value in enumerate(params):
        expression = expression.replace(f'%{index}', value)
    intension = Node()
    intension.from_string_rec(expression)
    intension.reduce_logic(True)
    res = intension.normalize(True)
    var_idx = []
    coeffs = []
    b = []
    comp = []
        
    for linear_cons in res:
        new_var, new_coeffs, new_b, new_comp = Node.get_arrays_from_dict(linear_cons, parser)
        var_idx.append(new_var)
        coeffs.append(new_coeffs)
        b.append(new_b)
        comp.append(new_comp)
    
    return var_idx, coeffs, b, comp