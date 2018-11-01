import numpy as np
import pickle

Xd = pickle.load(open("RML2016.10a_dict.dat", 'rb'), encoding='Latin-1')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
modul = ['QAM16', 'QAM64']
with open("QAMDataset.txt", "w") as f:
    for mod in modul:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
            for s in Xd[(mod, snr)]:
                m = len(s)
                p1 = 0
                p2 = 0
                for j in range(len(s[0])):
                    p1+= (s[0][j])**2
                    p2+= (s[1][j])**2
                pw = (1/m)*(p1 + p2)
                f.write("\n" + 'Modulation_type :' + str(mod) + "\n" + 'SNR_Value :' + str(snr) + "\n"+ 'Power_Value :' + str(pw) + "\n" + str(s) + "\n" )

f.close()

X = np.vstack(X)
n_example = X.shape[0]
print('Number of signals in input :', n_example)