from common import gen_traces_serial,sbox
import numpy as np
from stella.preprocessing import SNR,SNROrder
from stella.utils import DataReader
from stella.attacks.sasca import create_graph,PROFILE
import pickle

# Setup the simulation settings
D=2
tag = "sim"
DIR_PROFILE = "./traces/profile/"
DIR_ATTACK = "./traces/attack/"
nfile_profile = 10
nfile_attack = 1
ntraces = 10000
std = 1
fgraph = "./graph.txt"
## Generate the profiling and attack sets
gen_traces_serial(nfile_profile,
        ntraces,std,tag,
        DIR_PROFILE,random_key=True,D=D)
secret_key = gen_traces_serial(nfile_attack,ntraces,
                    std,tag,DIR_ATTACK,random_key=False,D=D)[0]

## generate the graph
print("Generate the graph")
with open(fgraph, 'w') as fp:
    for i in range(16): fp.write("k%d #secret \n"%(i))
    fp.write("sbox #table \n")
    fp.write("\n\n#indeploop\n\n")
    for i in range(16): fp.write("p%d #public \n"%(i))
    for i in range(16): fp.write("y%d = k%d + p%d \n"%(i,i,i))
    for i in range(16): fp.write("x%d = y%d -> sbox \n"%(i,i))
    for i in range(16):
        for d in range(D): fp.write("x%d_%d #profile \n"%(i,d))
        for d in range(D): fp.write("y%d_%d #profile \n"%(i,d))
        add = ' ^ '.join(["x%d_%d"%(i,d) for d in range(D)])
        fp.write("x%d = "%(i)+add+"\n")
        add = ' ^ '.join(["y%d_%d"%(i,d) for d in range(D)])
        fp.write("y%d = "%(i)+add+"\n")
    fp.write("\n\n#endindeploop\n\n")

graph = create_graph(fgraph)
pickle.dump(graph,open("graph.pkl",'wb'))

print("Start Profiling")
# File for profiling
files_traces = [DIR_PROFILE+"/traces/"+tag+"_traces_%d.npy"%(i) for i in range(nfile_profile)]
files_labels = [DIR_PROFILE+"/labels/"+tag+"_labels_%d.npz"%(i) for i in range(nfile_profile)]

graph = pickle.load(open("graph.pkl","rb"))
profile_var = {}
for var in graph["var"]:
    if graph["var"][var]["flags"] & PROFILE != 0:
        profile_var[var] = {}

#Go over all the profiling traces and update the SNROrder 
for it,(traces,labels) in enumerate(zip(DataReader(files_traces,None),DataReader(files_labels,["labels"]))):
    labels = list(filter(lambda x: x["label"] in profile_var,labels[0]))
    if it == 0:
        ntraces,Ns = traces.shape
        snr = SNR(Np=len(labels),Nc=256,Ns=Ns)
        data = np.zeros((len(labels),ntraces))

    for i,l in enumerate(labels):
        data[i,:] = l["value"]
        profile_var[l["label"]]["offset"] = i
    snr_val = snr.fit_u(traces,data)

for v in profile_var:
    profile_var[v]["SNR"] = snr_val[profile_var[v]["offset"],:]
    print("#",v,"->",np.max(profile_var[v]["SNR"]))


print("Start Attack")
# Attack files
files_traces = [DIR_ATTACK+"/traces/"+tag+"_traces_%d.npy"%(i) for i in range(nfile_attack)]
files_labels = [DIR_ATTACK+"/labels/"+tag+"_labels_%d.npz"%(i) for i in range(nfile_attack)]

# For each attack file
for it,(traces,labels) in enumerate(zip(DataReader(files_traces,None),DataReader(files_labels,["labels"]))):
    if it == 0:
        ntraces,Ns = traces.shape
        mcp_dpas = [MCP_DPA(SM[i,:,:],
                            u[i,:,:],
                            s[i,:,:],256,D) for i in range(16)]
        
        # Memory for every guess at the output of the Sbox
        guesses = np.zeros((256,ntraces)).astype(np.uint16)
        # generate all the key guess
        kg = np.repeat(np.arange(256),ntraces).reshape(256,ntraces).astype(np.uint16)

    for i,mcp_dpa in enumerate(mcp_dpas):
        # get a plaintext byte value
        data = next(filter(lambda x:x["label"]=="p%d"%(i),labels[0]))["value"]
        # derive the Sbox output
        guesses[:,:] = sbox[data^kg]
        # update correlation
        mcp_dpa.fit_u(traces,guesses)

# Display the obtained key
for i,k in enumerate(secret_key):
    guess = np.argmax(np.max(np.abs(mcp_dpas[i]._corr),axis=1))
    print("Key :",k,"guess: ",guess)


