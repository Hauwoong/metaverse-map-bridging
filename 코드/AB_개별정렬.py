"""A 와 B 를 각각 따로 합본에 맞춰, 두 변환이 얼마나 다른지 잰다.

각각은 잘 맞는데 둘이 다르면 → 재구성은 멀쩡하고 **A↔B 상대 위치만** 틀린 것.
"""
import os, numpy as np
H=os.path.expanduser("~")
def q2R(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
def rd(p):
    out={}; L=[l for l in open(p) if not l.startswith("#")]
    for i in range(0,len(L),2):
        t=L[i].split()
        if len(t)<10 or not t[9].endswith(".png"): continue
        R=q2R(list(map(float,t[1:5]))); tv=np.array(list(map(float,t[5:8])))
        out[t[9]]=dict(R=R,C=-R.T@tv)
    return out
def proj(M):
    U,_,Vt=np.linalg.svd(M); D=np.eye(3); D[2,2]=np.sign(np.linalg.det(U@Vt)); return U@D@Vt
def fit(S,M,ns):
    Mr=np.zeros((3,3))
    for n in ns: Mr += M[n]["R"].T @ S[n]["R"]
    R=proj(Mr)
    Ps=np.array([S[n]["C"] for n in ns]); Pt=np.array([M[n]["C"] for n in ns])
    cs,ct=Ps.mean(0),Pt.mean(0); Xs,Xt=Ps-cs,Pt-ct
    den=float((Xs*Xs).sum()); s=float((Xt*(Xs@R.T)).sum())/den if den>1e-12 else 1.0
    t=ct-s*(R@cs)
    res=np.array([np.linalg.norm(M[n]["C"]-(s*(R@S[n]["C"])+t)) for n in ns])
    return s,R,t,res
def trim(S,M,ns,rounds=6,keep=0.85):
    cur=list(ns)
    for _ in range(rounds):
        if len(cur)<8: break
        s,R,t,res=fit(S,M,cur)
        k=max(8,int(len(cur)*keep))
        if k>=len(cur): break
        cur=[cur[i] for i in np.argsort(res)[:k]]
    s,R,t,res=fit(S,M,cur)
    return s,R,t,res,cur

B=open(H+"/data/nogt/wp4/best.txt").read().strip()
S=rd(B+"/images.txt"); M=rd(H+"/data/sfm_all/sparse/1/images.txt")
def ix(k): return int(k.split("_")[1].split(".")[0])
A=[k for k in S if k in M and k.startswith("v1791_") and ix(k)<=143]
Bl=[k for k in S if k in M and k.startswith("v1793_")]
print("%-6s %6s %8s %10s %10s"%("장면","사용","스케일","잔차중앙","잔차95%"))
print("-"*46)
R_={}
for lab,L in (("A",A),("B",Bl)):
    s,R,t,res,cur=trim(S,M,L)
    R_[lab]=(s,R,t)
    print("%-6s %4d/%-3d %8.4f %10.4f %10.4f"%(lab,len(cur),len(L),s,np.median(res),np.percentile(res,95)))
sa,Ra,ta=R_["A"]; sb,Rb,tb=R_["B"]
print()
print("두 변환의 차이")
print("  스케일 비       %.3f  (1.0 이면 같음)"%(sb/sa))
ang=np.degrees(np.arccos(np.clip((np.trace(Ra.T@Rb)-1)/2,-1,1)))
print("  회전 차이       %.1f°"%ang)
# A 변환으로 B 중심을 옮겨보면 합본 B 와 얼마나 떨어지나
d=[np.linalg.norm(M[n]["C"]-(sa*(Ra@S[n]["C"])+ta)) for n in Bl]
sz=float(np.linalg.norm(np.array([M[n]["C"] for n in Bl])-np.array([M[n]["C"] for n in Bl]).mean(0),axis=1).mean())
print("  A 변환을 B 에 적용 시 어긋남  중앙 %.3f (B 크기 %.3f 의 %.0f%%)"%(np.median(d),sz,100*np.median(d)/sz))
