import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
#import cmath
from scipy.io import loadmat
Weight=loadmat("Weight.mat")
Weightnew=Weight["Weight"]
Base=loadmat("Base.mat")
Basenew=Base["Base"]
C=loadmat("C.mat")
Cnew=C["C"]


def Unit_Disk(dist=0.02):
    t=[]
    t=np.linspace(-1,1,101,dtype=np.float64)
    #print(t)
    real=t
    image=np.float64(t)
    n=len(t)
    ret1=np.zeros((n,n),dtype=np.complex128)
    for i in range(n):
        for k in range(n):
            ret1[i][k]=complex(real[i],image[k])    
    ret1=ret1.reshape((n**2,1))
    ret2=np.zeros((ret1.shape[0],1),dtype=np.complex128)
    ret3=[]
    for i in range(len(ret1)):
        if abs(ret1[i]) < 0.99:
            ret2[i]=1
        else:
            ret2[i]=0 
    ret2=ret1*ret2
    ret2=ret2.reshape(len(ret2),).tolist()
    #print(ret2)
    #retnew=[i for i in sorted(enumerate(ret2), key=lambda x:np.sqrt(x.real**2+x.imag**2))]
    ret2 = sorted(ret2, key=lambda x: (np.sqrt(x.real**2+x.imag**2),cmath.phase(x)),reverse=False)
    #print(ret2)
    #for i in enumerate(ret2):
    #    print(i)
    for i in range(len(ret2)):
        if abs(ret2[i]) > 0:
            ret3.append(ret2[i])
    ret3.append(0) 
    #print(ret3)
    return ret3

def weight(n,Order=3):
    y=np.zeros((n,1))
    Newton=np.zeros((Order+1,Order))
    Newton[0:2,0]=[1/2,1/2]
    Newton[0:3,1]=[1/6,4/6,1/6]
    Newton[0:4,2]=[1/8,3/8,3/8,1/8]
    Newton[0:5,3]=[7/90,16/12,2/15,16/45,7/90]
    Newton[0:6,4]=[19/288,25/96,25/144,25/144,25/96,19/288]
    Newton[0:7,5]=[41/840,9/35,9/280,34/105,9/280,9/35,41/840]
    k=(n-1)//Order
    y=np.array(y)
    Newton=np.array(Newton)

    if k>0:
        for i in range(k):
            y[(i*Order):((i+1)*Order+1)]=y[(i*Order):((i+1)*Order+1)]+Newton[:,Order-1].reshape(len(Newton[:,Order-1]),1)

    y=y*Order/(n-1)
    nleft=n-k*Order-1
    
    if nleft>0:
        y[(n-nleft-1):]=y[(n-nleft-1):]+Newton[0:(nleft+1),nleft-1].reshape(len(y[(n-nleft-1):]),1)*nleft/(n-1)
    return y

    
def intg(f,g):
    Weigth=weight(len(f),6)
    f=np.array(f)
    g=np.array(g)
    f=f.reshape(len(f),1)
    g=g.reshape(len(g),1)
    Weigth=np.array(Weigth)
    #print(Weigth.shape)
    #print(np.transpose(g*Weigth).shape)
    Wret=np.dot(np.transpose(g*Weigth),f)
    #print(Wret)
    return Wret

def e_a(a,z):
    a=np.array(a)
    z=np.array(z)
    ret=(np.sqrt((1-abs(a)**2)))/(1-a.conjugate()*z)
    return ret


def AFD_test(f,n=50,Basenew=Basenew,Weightnew=Weightnew,tol=1e-18):
    t=np.linspace(0,2*math.pi,len(f))
    m=len(f)
    G=np.zeros((n,m),dtype=complex)
    a=np.zeros((n,1),dtype=complex)
    G[0,:]=f
    C=Unit_Disk()
    N=len(C)
    f2=intg(f,f)
    Weight=weight(len(f),6)
    Base=np.zeros((len(C),m),dtype=np.complex64)
    temp=[cmath.exp(x*complex(0,1)) for x in t]

    for i in range(len(Cnew)):
        Base[i,:]=e_a(Cnew[i],temp)
    #print(Base)
    coef=np.zeros((n,1),dtype=np.complex64)
    S1=np.zeros(np.array(C).shape)
    tem_B=1
    temp=[cmath.exp(x*complex(0,1)) for x in t]
    F=np.zeros(G.shape)
    j=0
    fn=0
    
    coef[0,:]=intg(f,np.ones(len(t)))
    a[j]=0
    tem_B=(np.sqrt(1-abs(a[j]**2)))/(1-a[j].conjugate()*temp)*tem_B
    F[j,:]=coef[j]*tem_B
    fn=fn+F[j,:]
    err=10
    f_recovery=np.zeros((len(f),1))
    Snew=[]
    #print("f:",f)
    while err>=tol and j<n-1:
            j=j+1
            G[j,:]=((G[j-1,:])-coef[j-1]*e_a(a[j-1],temp))*(1-a[j-1].conjugate()*temp)/(temp-a[j-1])
            I=0;
            S1=(np.mat(Base)*np.mat((np.array(G[j,:]).reshape(len(G[j,:]),1).conjugate()*Weight))).conjugate()
            #S1=(np.mat(Base)*np.mat((np.array(G[j,:]).reshape(len(G[j,:]),1).conjugate()*Weightnew))).conjugate()
            S2=np.zeros(len(S1))
            S2=abs(S1)
            S2=S2.tolist()
            I=S2.index(max(S2))
            #print(I)
            coef[j]=S1[I]
            a[j]=Cnew[I]
            tem_B=(np.sqrt(1-abs(a[j])**2)/(1-(a[j].conjugate())*temp))*((temp-a[j-1])/np.sqrt(1-abs(a[j-1])**2))*tem_B
            #print(tem)
            F[j,:]=coef[j]*tem_B
            fn=fn+F[j,:]
            Snew.append(a[j])
            err=abs(intg(2*np.real(fn)-coef[0]-f,2*np.real(fn)-coef[0]-f))/f2
    f_recovery=2*np.real(fn)-coef[0]
    #f_recovery=2*np.real(fn)-coef[0]
    #print("Snew:",Snew)
    return Snew


