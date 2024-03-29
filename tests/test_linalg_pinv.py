from ypcutil.importcuda import *
import ypcutil.linalg as sl

G=np.random.rand(200,100)
d_G = parray.to_gpu(G)
Ginv = np.linalg.pinv(G, 1e-4)
d_Ginv = sl.pinv(G)
print (np.abs(Ginv.real-d_Ginv.get().real).max())
print (np.abs(Ginv.imag-d_Ginv.get().imag).max())
print np.abs(sl.dot(sl.dot(d_G, d_Ginv),d_G).get()-G).max()

G=np.random.rand(100,200)
d_G = parray.to_gpu(G)
Ginv = np.linalg.pinv(G, 1e-4)
d_Ginv = sl.pinv(G)
print (np.abs(Ginv.real-d_Ginv.get().real).max())
print (np.abs(Ginv.imag-d_Ginv.get().imag).max())
print np.abs(sl.dot(sl.dot(d_G, d_Ginv),d_G).get()-G).max()

G=np.random.rand(200,200)
d_G = parray.to_gpu(G)
Ginv = np.linalg.pinv(G, 1e-4)
d_Ginv = sl.pinv(G)
print (np.abs(Ginv.real-d_Ginv.get().real).max())
print (np.abs(Ginv.imag-d_Ginv.get().imag).max())
print np.abs(sl.dot(sl.dot(d_G, d_Ginv),d_G).get()-G).max()

G=np.random.rand(200,100) + 1j*np.random.rand(200,100)
d_G = parray.to_gpu(G)
Ginv = np.linalg.pinv(G, 1e-4)
d_Ginv = sl.pinv(G)
print (np.abs(Ginv.real-d_Ginv.get().real).max())
print (np.abs(Ginv.imag-d_Ginv.get().imag).max())
print np.abs(sl.dot(sl.dot(d_G, d_Ginv),d_G).get()-G).max()

G=np.random.rand(100,200)+1j*np.random.rand(100,200)
d_G = parray.to_gpu(G)
Ginv = np.linalg.pinv(G, 1e-4)
d_Ginv = sl.pinv(G)
print (np.abs(Ginv.real-d_Ginv.get().real).max())
print (np.abs(Ginv.imag-d_Ginv.get().imag).max())
print np.abs(sl.dot(sl.dot(d_G, d_Ginv),d_G).get()-G).max()

G=np.random.rand(200,200)+1j*np.random.rand(200,200)
d_G = parray.to_gpu(G)
Ginv = np.linalg.pinv(G, 1e-4)
d_Ginv = sl.pinv(G)
print (np.abs(Ginv.real-d_Ginv.get().real).max())
print (np.abs(Ginv.imag-d_Ginv.get().imag).max())
print np.abs(sl.dot(sl.dot(d_G, d_Ginv),d_G).get()-G).max()
