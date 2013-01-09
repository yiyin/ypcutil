from importcuda import *
import linalg as sl
G=np.random.rand(100,200)+1j*np.random.rand(100,200)
d_G = parray.to_gpu(G)
S=sl.svd(G, compute_v=0, compute_u=0)
u,s,v=np.linalg.svd(G)
print "s alone: %e" % (np.abs(s-S.get()).max())


compute_u = 1
compute_v = 0
econ = 0
d_G = parray.to_gpu(G)
U,S=sl.svd(G, compute_u=compute_u, compute_v=compute_v, econ = econ)
print "s,u=%d,v=%d,econ=%d: %e %e" % (compute_u, compute_v, econ,np.abs(s-S.get()).max(),np.abs(np.abs(u.imag)-np.abs(U.get().imag)).max())

compute_u = 0
compute_v = 1
econ = 0
d_G = parray.to_gpu(G)
S,V=sl.svd(G, compute_u=compute_u, compute_v=compute_v, econ = econ)
print "s,u=%d,v=%d,econ=%d: %e %e" % (compute_u, compute_v, econ,np.abs(s-S.get()).max(),np.abs(np.abs(v.imag)-np.abs(V.get().imag)).max())

compute_u = 1
compute_v = 1
econ = 0
d_G = parray.to_gpu(G)
U,S,V=sl.svd(G, compute_u=compute_u, compute_v=compute_v, econ = econ)
print "s,u=%d,v=%d,econ=%d: %e %e %e" % (compute_u, compute_v, econ,np.abs(s-S.get()).max(),np.abs(np.abs(u.imag)-np.abs(U.get().imag)).max(),np.abs(np.abs(v)-np.abs(V.get())).max())


compute_u = 1
compute_v = 0
econ = 1
d_G = parray.to_gpu(G)
U,S=sl.svd(G, compute_u=compute_u, compute_v=compute_v, econ = econ)
print "s,u=%d,v=%d,econ=%d: %e %e" % (compute_u, compute_v, econ,np.abs(s-S.get()).max(),np.abs(np.abs(u[:,0:min(G.shape)].imag)-np.abs(U.get().imag)).max())

compute_u = 0
compute_v = 1
econ = 1
d_G = parray.to_gpu(G)
S,V=sl.svd(G, compute_u=compute_u, compute_v=compute_v, econ = econ)
print "s,u=%d,v=%d,econ=%d: %e %e" % (compute_u, compute_v, econ,np.abs(s-S.get()).max(),np.abs(np.abs(v[0:min(G.shape),:].imag)-np.abs(V.get().imag)).max())

compute_u = 1
compute_v = 1
econ = 1
d_G = parray.to_gpu(G)
U,S,V=sl.svd(G, compute_u=compute_u, compute_v=compute_v, econ = econ)
print "s,u=%d,v=%d,econ=%d: %e %e %e" % (compute_u, compute_v, econ,np.abs(s-S.get()).max(),np.abs(np.abs(u[:,0:min(G.shape)].imag)-np.abs(U.get().imag)).max(),np.abs(np.abs(v[0:min(G.shape),:])-np.abs(V.get())).max())
