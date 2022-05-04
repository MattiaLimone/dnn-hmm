from spafe.utils import vis
import preprocessing.utils as utl
from spafe.features.lpc import lpc, lpcc

# init input vars
num_ceps = 13
lifter = 0
normalize = True

# read wav
sig, fs = utl.remove_silence('data/lisa/data/timit/raw/TIMIT/TEST/DR1/FAKS0/SA1.WAV')

# compute lpcs
lpcs = lpc(sig=sig, fs=fs, num_ceps=num_ceps)
# visualize features
vis.visualize_features(lpcs, 'LPC Index', 'Frame Index')


# visualize spectogram
vis.spectogram(sig, fs)
# compute lpccs
lpccs = lpcc(sig=sig, fs=fs, num_ceps=num_ceps, lifter=lifter, normalize=normalize)
# visualize features
vis.visualize_features(lpccs, 'LPCC Index', 'Frame Index')

print('LPC')
print(lpcs)
print('\n')
print(lpcs[0])
print('\n')
print('LPCC')
print(lpccs)
print(len(lpccs[0]))



