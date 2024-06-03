import scipy.io
mat = scipy.io.loadmat('ptb.mat')
data = mat['data_to_save']
train_data = data[0,0]
test_data = data[0,1]
raw_data = train_data[:,0]
bpms = train_data[:,1]
locs = train_data[:,2]
print('exit')
