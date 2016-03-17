
import numpy as np
import scipy as sc
from numpy.linalg import inv
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from scipy import linalg
from numpy.random import rand
#from scipy.linalg import inv

print("numpy testing")
a = np.array([[1,2,3,4,5],[2,3,7,10,13],[3,5,11,16,21], [2,-7,7,7,2],[1,4,5,3,10]])
b = np.array([2,12,12,57,7])
x1=np.linalg.solve(a,b)
print(x1)
print("linalg check...")
if(np.allclose(np.dot(a,x1),b)==True):
    print("OK")
else:
    print("failed")
x2=np.dot(inv(a),b)
print(x2)
print("inv check...")
if(np.allclose(np.dot(a,x2),b)==True):
    print("OK")
else:
    print("failed")
print()
print("scipy testing")
x4=sc.linalg.solve(a,b)
print(x4)
print("linalg check...")
if(sc.allclose(sc.dot(a,x4),b)==True):
    print("OK")
else:
    print("failed")
x5=sc.dot(sc.linalg.inv(a),b)
print(x5)
print("inv check...")
if(sc.allclose(sc.dot(a,x5),b)==True):
    print("OK")
else:
    print("failed")

a3=np.loadtxt("A.dat")
b3=np.loadtxt("B.dat")
x3=np.linalg.solve(a3,b3)
print(x3)
print("reading data from file check...")
if(np.allclose(np.dot(a3,x3),b3)==True):
    print("OK")
else:
    print("failed")

a4=np.array([[4,1,0],[1,4,0],[-1,1,5]])
print("Eigenvalues and right eigenvectors")
eigvec=sc.linalg.eig(a4)
print(eigvec)
eigval=sc.linalg.eigvals(a4)
print("Eigenvalues")
print(eigval)
print("Eigenvalues and eigenvectors check...")
for i in range(len(eigvec[1])):
    if norm(np.dot(a4,eigvec[1][:,i][np.newaxis].T)-np.dot(eigval[i],eigvec[1][:,i][np.newaxis].T))>1e-10:
        print("failed")
        break
else:
    print("OK")

n=1000 #matrix size
a_sp=sc.sparse.rand(n,n,density=10./n,format='csr')
b_sp=sc.sparse.rand(n,1,density=1,format='csr')
a_sym=a_sp.dot(a_sp.transpose())
b_sym=rand(n)
x_sp=spsolve(a_sp,b_sp)
x_sym,info=sc.sparse.linalg.minres(a_sym,b_sym,tol=1e-18)
print("Sparse martix solve check...")
#print(a_sp.dot(x_sp.transpose())-b_sp)
if(norm(a_sp.dot(x_sp[np.newaxis].T)-b_sp)<1e-10):
    print("OK")
else:
    print("failed")
print("Sparse symmetric martix solve check...")
if(info==0 and norm(a_sym.dot(x_sym[np.newaxis].T)-b_sym[np.newaxis].T)<1e-3):
    print("OK")
else:
    print("failed")

#print(a_sp.toarray())
#print(b_sp)
eigvec_num=6
eigval_sp,eigvec_sp=sc.sparse.linalg.eigs(a_sp,eigvec_num)
#print(len(eigvec_sp))
print("Eigenvalues and eigenvectors of sparse matrix check ...")
for i in range(eigvec_num):
    #print(a_sp.dot(eigvec_sp[:,i][np.newaxis].T))
    if norm(a_sp.dot(eigvec_sp[:,i][np.newaxis].T)-np.dot(eigval_sp[i],eigvec_sp[:,i][np.newaxis].T))>1e-5:
        print("failed")
        break
else:
    print("OK")

eigval_sym,eigvec_sym=sc.sparse.linalg.eigsh(a_sym,eigvec_num)
#print(len(eigvec_sp))
print("Eigenvalues and eigenvectors of symmetric sparse matrix check ...")
for i in range(eigvec_num):
    #print(a_sp.dot(eigvec_sp[:,i][np.newaxis].T))
    if norm(a_sym.dot(eigvec_sym[:,i][np.newaxis].T)-np.dot(eigval_sym[i],eigvec_sym[:,i][np.newaxis].T))>1e-5:
        print("failed")
        break
else:
    print("OK")
a_sym_arr=a_sym.toarray()
array_for_im=[]
for i in range(len(a_sym_arr)):
    tmp=[]
    for j in range(len(a_sym_arr[i])):
        if(a_sym_arr[i][j]!=0):
            tmp.append(1)
        else:
            tmp.append(0)
    array_for_im.append(tmp)

sc.misc.imsave('outfile.bmp', array_for_im)