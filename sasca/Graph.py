import numpy as np
from tqdm import tqdm
import networkx as nx
import ctypes

############################
#     Function for nodes
############################ 
bxor = np.bitwise_xor   #ID 2
band = np.bitwise_and   #ID 0
binv = np.invert        #ID 1
def ROL16(a, offset):   #ID 3
    Nk = 2**16
    a = a
    if offset == 0:
        return a
    rs = int(np.log2(Nk) - offset)
    return  (((a) << offset) ^ (a >> (rs)))%Nk

##############################
distribution_dtype = np.double
all_functions = [band,binv,bxor,ROL16]

class Graph():
    @staticmethod
    def wrap_function(lib, funcname, restype, argtypes):
        """Simplify wrapping ctypes functions"""
        func = lib.__getattr__(funcname)
        func.restype = restype
        func.argtypes = argtypes
        return func

    def __init__(self,Nk,nthread=8,vnodes=None,fnodes=None,DIR=""):
        if vnodes is None:
            vnodes = VNode.buff
        self._vnodes = vnodes

        if fnodes is None:
            fnodes = FNode.buff
        self._fnodes = fnodes

        self._nthread = nthread
        self._Nk = Nk

        self._fnodes_array = (FNode*len(fnodes))()
        self._vnodes_array = (VNode*len(vnodes))()

        for i,node in enumerate(fnodes):
            self._fnodes_array[i] = node

        for i,node in enumerate(vnodes):
            self._vnodes_array[i] = node

        self._lib = ctypes.CDLL(DIR+"./libbp.so")
        self._run_bp = self.wrap_function(self._lib,"run_bp",None,[ctypes.POINTER(VNode),
                ctypes.POINTER(FNode),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32])
    def run_bp(self,it=1):
        self._run_bp(self._vnodes_array,
            self._fnodes_array,
            ctypes.c_uint32(self._Nk),
            ctypes.c_uint32(len(self._vnodes)),
            ctypes.c_uint32(len(self._fnodes)),
            ctypes.c_uint32(it),
            ctypes.c_uint32(self._nthread))
class VNode(ctypes.Structure):
    """
        This object contains the variable nodes of the factor graph

        It contains multiple variables:
            - value: is the value of the actual node. This can be a scallar of a
            numpy array (do to // computationon the factor graph)
            - id: is the id of then node. The ids are distributed in ordre
            - result_of: is the function node that outputs this variable node
            - used_by: is the function node that use this variable node

    """
    N = 0
    buff = []
    _fields_ = [('id', ctypes.c_uint32),
            ('Ni', ctypes.c_uint32),
            ('Nf', ctypes.c_uint32),
            ('Ns', ctypes.c_uint32),
            ('update', ctypes.c_uint32),
            ('relative', ctypes.POINTER(ctypes.c_uint32)),
            ('id_input', ctypes.c_uint32),
            ('id_output', ctypes.POINTER(ctypes.c_uint32)),
            ('msg', ctypes.POINTER(ctypes.c_double)),
            ('distri_orig', ctypes.POINTER(ctypes.c_double)),
            ('distri', ctypes.POINTER(ctypes.c_double))] 
    @staticmethod
    def reset_all():
        for b in VNode.buff:
            del b
        VNode.buff = []
        N = 0

    def __init__(self,value=None,result_of=None):
        """
            value: is the value of the node
            result_of: is the function node that output this variable
                (None if no function is involved)
        """
        self._value = value
        self._result_of = result_of
        self._id = VNode.N
        VNode.N += 1
        VNode.buff.append(self)
        self._flag = 0
        # say to the funciton node that this is its output. 
        if result_of is not None: 
            result_of.add_output(self)

        # all the function nodes taking self as input
        self._used_by = []

    def eval(self):
        """
            returns the value of this variable node. To do so, 
            search of the output of the parent node
        """
        if self._value is None:
            self._value = self._result_of.eval()
        return self._value

    def used_by(self,fnode):
        """
            add the fnode to the list of fnodes using this variable
        """
        self._used_by.append(fnode)
    def __str__(self):
        return "v" + str(self._id)

    def initialize(self,Nk=None,distri=None):
        """ Initialize the variable node. It goes in all its neighboors and
            searchs for its relative position with their lists

            args:
                - distri: the initial distribution of the node
                - Nk: the number of possible values that this node can take
            the two cannot be None at the same time ...

            created state:
                - relative contains the position of this variable in its functions nodes
                - distri extrinsic distribution of the node
                - distri_orig intrinsic distriution of the node
                - id_neighboor: if of the neighboors, starting with the result_of
        """
        if Nk is None and distri is None:
            raise Exception("Nk and distri cannot be None at the same time")

        if distri is None:
            distri = np.ones(Nk,dtype=distribution_dtype)/Nk
        else:
            distri = distri.astype(distribution_dtype)
            Nk = len(distri)

        # header
        self.id = np.uint32(self._id)
        self.Ni = np.uint32(self._result_of is not None)
        self.Nf = np.uint32(len(self._used_by))
        self.Ns = np.uint32(Nk)
        self.update = np.uint32(1)
        
        # relative contains the position of this variable node
        # at in input of each of the functions that use it. In fnodes, 
        # the msg with index 0 is always the output. There comes the 1+. 
        self._relative = np.array([1+fnode._inputs.index(self) for fnode in self._used_by]).astype(np.uint32)
        self._distri = distri.astype(dtype=distribution_dtype)
        self._distri_orig = self._distri.copy()

        nmsg = self.Ni + self.Nf
        # one message to result_of and on to each function using this node
        self._msg = np.zeros((nmsg,Nk),dtype=distribution_dtype)
        for i in range(nmsg):
            self._msg[i,:] = distri
        
        # function node that outputs this node
        if self.Ni > 0:
            self._id_input = np.uint32(self._result_of._id)
        else:
            self._id_input = np.uint32(0)

        # function node that uses this node
        if self.Nf > 0:
            self._id_output = np.array([node._id for node in self._used_by],dtype=np.uint32)
        else:
            self._id_output = np.array([],dtype=np.uint32)
        
        tmp = []
        if self._result_of is not None:
            tmp.append(self._result_of._id)
        for node in self._used_by:
            tmp.append(node._id)
        self._id_neighboor = np.array(tmp,dtype=np.uint32)
        
        self.relative = self._relative.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.id_output = self._id_output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.id_input = self._id_input
        self.msg = self._msg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.distri_orig = self._distri_orig.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.distri = self._distri.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

class FNode(ctypes.Structure):
    """
        This object contains the function nodes of the factor graph

        It contains multiple variables:
            - func: is the function that is applyed to the the node. All the
              available function are above in this file
            - id: is the id of then node. The ids are distributed in ordre
            - inputs: are the variable nodes at the input of this function
            - output: is the ouput of this function node.
    """
    _fields_ = [('id', ctypes.c_uint32),
            ('li', ctypes.c_uint32),
            ('has_offset', ctypes.c_uint32),
            ('offset', ctypes.c_uint32),
            ('func_id', ctypes.c_uint32),
            ('i', ctypes.POINTER(ctypes.c_uint32)),
            ('o', ctypes.c_uint32),
            ('relative', ctypes.POINTER(ctypes.c_uint32)),
            ('msg', ctypes.POINTER(ctypes.c_double))] 

    N = 0
    buff = []
    @staticmethod
    def reset_all():
        for b in FNode.buff:
            del b
        FNode.buff = []
        N = 0

    def __init__(self,func,inputs=None,offset=None):
        """
            func: the function implemented by the nodes
            input: a list with the input variable nodes that are the 
            inputs of this node
            offset: is the constant second argument of func
        """

        #add this node the the list
        self._id = FNode.N
        FNode.N +=1
        FNode.buff.append(self)

        self._func = func
        self._func_id = all_functions.index(func)
        self._inputs = inputs
        if offset is None:
            self._has_offset = False
            self._offset = np.uint32(0)
        else:
            self._has_offset = True
            self._offset = np.uint32(offset)

        # notify that all the inputs that they are used here
        if inputs is not None:
            for n in inputs:
                n.used_by(self)

    def __str__(self):
        return "f" + str(self._id)

    def eval(self):
        """
            apply the function to its inputs and return 
            the output
        """
        I = []
        for v in self._inputs:
            I.append(v.eval())

        if len(I) == 1:
            if self._has_offset:
                return self._func(I[0],self._offset)
            else:
                return self._func(I[0])
        else:
            return self._func(I[0],I[1])

    def add_output(self,vnode):
        self._output = vnode
    
    def initialize(self,Nk):
        """ initialize the message memory for this function node"""
        nmsg = len(self._inputs) + 1

        ## Position of the inputs in the variable nodes. 
        # The output node is always first in the variable node
        self._i = np.array([node._id for node in self._inputs]).astype(np.uint32)
        self._o = np.uint32(self._output._id)
        self._relative = np.array([np.where(vnode._id_neighboor==self._id)[0] for vnode in self._inputs]).astype(np.uint32)
        self._msg = np.zeros((nmsg,Nk),dtype=distribution_dtype)
        self._indexes = np.zeros((3,Nk),dtype=np.uint32)
        for i in range(3):
            self._indexes[i,:] = np.arange(Nk)

        self.id = np.uint32(self._id)
        self.li = np.uint32(len(self._i))
        self.has_offset = np.uint32(self._has_offset)
        self.offset = np.uint32(self._offset)
        self.func_id = np.uint32(self._func_id)

        self.i = self._i.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.o = np.uint32(self._o)
        self.relative = self._relative.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.msg = self._msg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def apply_func(func=bxor,inputs=[None],offset=None):
    """ apply the functionc func to the inputs and 
        returns the output node 
    """
    FTMP = FNode(func=func,inputs=inputs,offset=offset)
    return VNode(result_of=FTMP)

def initialize_graph(distri=None,Nk=None):
    """
        initialize the complete factor graph
        distri: (#ofVnode,Nk) or None. The row of distri are assigned to 
            the VNode with the row index
        Nk: the number of possible values for the variable nodes
    """
    for p,node in enumerate(VNode.buff):
        if distri is not None:
            d = distri[p,:]
            Nk = len(d)
        else:
            d = None
        node.initialize(distri=d,Nk=Nk)
    for node in FNode.buff:
        node.initialize(Nk=Nk)
def build_nx_grah(fnodes):
    G = nx.DiGraph()
    off = 0
    for F in fnodes:
        for vnode in F._inputs:
            G.add_edges_from([(vnode,F)])
        G.add_edges_from([(F,F._output)])
    return G

def plot_graph(fnodes=None):
    if fnodes is None:
        fnodes = FNode.buff
    G = build_nx_grah(fnodes)
    color_map=[]
    for node in G.nodes:
        if isinstance(node,VNode):
            color_map.append('r')
        else:
            color_map.append('g')
    nx.draw(G,with_labels=True,node_color=color_map)

def longest_path(fnodes):
    G = build_nx_graph(fnodes)
    return nx.algorithms.dag.dag_longest_path(G)
