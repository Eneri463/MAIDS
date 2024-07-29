import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

# ----------------------------------------------------------------------------
# задаём значение градиентов
def initValuesG():
    
    # ----- градиенты: шаг 1
    dF = [np.array([[1/10, 0],[0,0]]), np.array([[0, 0],[0,0]])]
    dPsi = [np.array([[0], [0]]), np.array([[0], [1]])]
    dG = [np.array([[0], [0]]), np.array([[0], [0]])]
    dH = [np.array([[0,0]]), np.array([[0,0]])]
    dQ = [0, 0]
    dR = [0, 0]
    dx0 =  [np.array([[0], [0]]), np.array([[0], [0]])] 
    dP0 = [np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])]
    
    return dF, dPsi, dG, dH, dQ, dR, dx0, dP0

# ----------------------------------------------------------------------------
# задаём начальные значения системы
def initValues(number, theta = np.array([-0.8, 1])):
    
    N = 8 # размер вектора Ui
    r = 1 # размер вектора u
    s = 2 # количество неизвестных параметров
    q = int((s*(s+1))/2+1) # кол-во точек в плане (кол-во Ui)
    
    if number == 1:
        return r, N, q, s
    
    n = 2 # размер вектора x
    m = 1 # размер вектора y
    
    F =  np.array([[theta[0]/10,-1],[0.1,0.1]])   # матрица процесса системы (матрица перехода)
    psi =  np.array([[0], [theta[1]]])         # матрица управления
    G =  np.array([[1], [1]])
    P0 = np.array([[0.1,0],[0,0.1]])
    Q =  np.array([[0.1]])
    R = np.array([[0.1]])
    H = np.array([[1,0]])
    x0 = np.zeros((n,1))
    
    if number == 2:
        return F, psi, G, H, Q, R, x0, P0, s, n, r, N, m

# ----------------------------------------------------------------------------
# A*B*A^T, где A,B - матрицы
def ABAt(A,B):
    res = np.dot(np.dot(A,B),A.transpose())
    return res   

# ----------------------------------------------------------------------------
# нахождение матрицы Ci
def cI(i, s, n):
    
    res = np.zeros((n, n*(s+1)))
    
    for k in range(n):
        res[k, k+n*i] = 1

    return res

# ----------------------------------------------------------------------------
# поиск информационной матрицы Фишера
def IMF(U):
    
    # ----- шаг 1
    
    F, psi, G, H, Q, R, x0, P0, s, n, r, N, m = initValues(2) 
    dF, dPsi, dG, dH, dQ, dR, dx0, dP0 = initValuesG()
    
    psiA = np.zeros((n*(s+1),r))
    
    for i in range(n):
        psiA[i][0] = psi[i][0]
    
    
    for i in range(s):
        for k in range(r):
            for j in range(n):
                psiA[i*n+n+j][k] = dPsi[i][j][k]
    
    # ----- шаг 2 
    Mtheta = np.zeros((s,s))
    
    Pkk = P0.copy()
    dPkk = dP0.copy()
    Bk = np.zeros((m,m)) # (B на шаге k)
    Kk = np.zeros((n,m)) # (K на шаге k)
    dKk = [ np.zeros((n,m)) for _ in range(s)] # (dK на шаге k)
    
    
    xA = np.zeros((n*(s+1),1))
    xAk1 = np.zeros((n*(s+1),1))
    eAk = np.zeros((n*(s+1),n*(s+1)))
    eAk1 = np.zeros((n*(s+1),n*(s+1)))

    for k in range(N):
        
        # шаг 3
        uk = np.expand_dims(U[k], axis=0).transpose()
        
        # ----- шаг 4
        if k!=0: 
            
            # ----- шаг 5

            # формула 26
            KTk = np.dot(F, Kk)
            
            dKTk = [ np.zeros((n,m)) for _ in range(s)]
            
            for i in range(s):
                dKTk[i] = np.dot(dF[i],Kk)+ np.dot(F, dKk[i])
            
            # ----- шаг 6
            
            # формула 23
            Fatk = np.zeros((n*(s+1),n*(s+1)))
            
            for i in range(n):
                for j in range(n):
                    Fatk[i][j] = F[i][j]
            
            # первый столбец
            for step in range(s):
                FatkTemp = dF[step] - np.dot(KTk, dH[step])
                for i in range(n):
                    for j in range(n):
                        Fatk[step*n+n+i][j] = FatkTemp[i][j]
            
            # диагонали
            FatkTemp = F - np.dot(KTk,H)
            for step in range(s):
                for i in range(n):
                    for j in range(n):
                        Fatk[step*n+n+i][step*n+n+j] = FatkTemp[i][j]
            
            
            # формула 25
            
            KAtk = np.zeros((n*(s+1),1))
            
            for i in range(n):
                KAtk[i][0] = KTk[i][0]
            
            for i in range(s):
                for j in range(n):
                    KAtk[i*n+n+j][0] = dKTk[i][j]

            # ----- шаг 7
            
            # формула 21 при k!=0
            xAk1 = np.dot(Fatk, xA) + np.dot(psiA, uk)
            
            # формула 22 при k!=0
            eAk1 = ABAt(Fatk, eAk) + ABAt(KAtk, Bk)
                    
            
        else:
            # формула 21 при k==0
            
            xAk1 = np.zeros((n*(s+1),1))
            xAk1Temp = np.dot(F,x0)+np.dot(psi,uk)
            
            for i in range(n):
                xAk1[i][0] = xAk1Temp[i][0]
            
            for i in range(s):
                xAk1Temp = np.dot(dF[i],x0)+np.dot(F,dx0[i])+np.dot(dPsi[i],uk)
                for j in range(n):
                    xAk1[i*n+n+j][0] = xAk1Temp[j][0]
            
            # формула 22 при k==0
            eAk1 = np.zeros((n*(s+1),n*(s+1)))
        
        # ----- шаг 8
        
        # ---------------------------------------------------------------------
        # вычисления, совпадающие с алгоритмами в лр 1
        
        
        # формула 10 
        Pk1k = ABAt(F,Pkk) + ABAt(G,Q)
        # формула 12 (B на шаге k+1)
        Bk1 = ABAt(H,Pk1k) + R
        # формула 13 (K на шаге k+1)
        Kk1 = np.dot(np.dot(Pk1k, H.transpose()), np.linalg.inv(Bk1))
        # формула 15 
        Pk1k1 = np.dot((np.eye(n)-np.dot(Kk1,H)),Pk1k)
        
        # ----- шаг 4 (лр 1, поиск градиента)
        dBk1 = [ np.zeros((m,m)) for _ in range(s)]
        dKk1 = [ np.zeros((n,m)) for _ in range(s)]
        dPk1k1 = [ np.zeros((n,n)) for _ in range(s)]
        
        for th in range(s):
            
            # +
            
            dPk1k = np.dot(np.dot(dF[th], Pkk),F.transpose())
            dPk1k += ABAt(F,dPkk[th]) + ABAt(G,dQ[th])
            dPk1k += np.dot(np.dot(F,Pkk),dF[th].transpose())
            dPk1k += np.dot(np.dot(dG[th],Q),G.transpose())
            dPk1k += np.dot(np.dot(G,Q), dG[th].transpose())
            
            # +
            dBk1[th] = np.dot((np.dot(dH[th], Pk1k)), H.transpose())
            dBk1[th] = dBk1[th] + ABAt(H, dPk1k)
            dBk1[th] = dBk1[th] + np.dot(np.dot(H,Pk1k),dH[th].transpose()) + dR[th]
            
            # +
            dKk1[th] = np.dot(dPk1k,H.transpose()) + np.dot(Pk1k,dH[th].transpose())
            dKk1[th] = dKk1[th] - np.dot(np.dot(np.dot(Pk1k,H.transpose()), np.linalg.inv(Bk1)), dBk1[th])
            dKk1[th] = np.dot(dKk1[th],np.linalg.inv(Bk1))
            
            # + 
            dPk1k1[th] = np.dot(np.eye(n)-np.dot(Kk1,H), dPk1k)
            dPk1k1[th] -= np.dot((np.dot(dKk1[th],H)+np.dot(Kk1,dH[th])), Pk1k)   
  
        
        # ---------------------------------------------------------------------
        
        # ----- шаг 9
        h1 = np.dot(xAk1, xAk1.T) + eAk1
                
        c0 = cI(0, s, n)
        c0T = c0.T
        invB = np.linalg.inv(Bk1)
        
        # формула 20
        for i in range(s):
            for j in range(s):
                
                h2 = np.dot(dH[i], c0)
                h2 = np.dot(h2,h1)
                h2 = np.dot(h2,c0T)
                h2 = np.dot(h2,dH[j].T)
                h2 = np.dot(h2, invB)
                
                h3 = np.dot(dH[i], c0)
                h3 = np.dot(h3,h1)
                h3 = np.dot(h3,(cI(j+1, s, n)).T)
                h3 = np.dot(h3,H.T)
                h3 = np.dot(h3, invB)
                
                h4 = np.dot(H, cI(i+1, s, n))
                h4 = np.dot(h4,h1)
                h4 = np.dot(h4,c0T)
                h4 = np.dot(h4,dH[j].T)
                h4 = np.dot(h4, invB)
                
                h5 = np.dot(H, cI(i+1, s, n))
                h5 = np.dot(h5,h1)
                h5 = np.dot(h5,(cI(j+1, s, n)).T)
                h5 = np.dot(h5,H.T)
                h5 = np.dot(h5, invB)
                
                h6 = np.dot(dBk1[i], np.linalg.inv(Bk1))
                h6 = np.dot(h6, dBk1[j])
                h6 = np.dot(h6, np.linalg.inv(Bk1))
                h6 = 0.5*h6
                
                Mtheta[i][j] += h2[0][0] + h3[0][0] + h4[0][0] + h5[0][0] + h6[0][0]
             
        Bk = Bk1.copy()
        Kk = Kk1.copy()
        Pkk = Pk1k1.copy()
        dKk = dKk1.copy()
        dPkk = dPk1k1.copy()
        xA = xAk1.copy()
        eAk = eAk1.copy()
                    
    return Mtheta

# -----------------------------------------------------------------------------
# производная информационной матрицы
def dIMF(U):
    
    # ----- шаг 1
    F, psi, G, H, Q, R, x0, P0, s, n, r, N, m = initValues(2)
    dF, dPsi, dG, dH, dQ, dR, dx0, dP0 = initValuesG() 
    
    # формула 24
    
    psiA = np.zeros((n*(s+1),r))
    
    for i in range(n):
        psiA[i][0] = psi[i][0]
    
    
    for i in range(s):
        for k in range(r):
            for j in range(n):
                psiA[i*n+n+j][k] = dPsi[i][j][k]
            
    # ----- шаг 2  
    
    dMtheta = [ [np.zeros((s,s)) for _ in range(N)] for _ in range(r)]
    
    Pkk = P0.copy()
    Kk = np.zeros((n,m)) # (K на шаге k)
    dKk = [ np.zeros((n,m)) for _ in range(s)] # (dK на шаге k)
    
    xA = np.zeros((n*(s+1),1))
    xAk1 = np.zeros((n*(s+1),1))
    
    dxAk = [[[np.zeros((n*(s+1),1)) for _ in range(N)] for _ in range(N)] for _ in range(r)]
    
    for k in range(N):
        
        # ----- шаг 3
        uk = np.expand_dims(U[k], axis=0).transpose()
        
        # ----- шаг 4 
        if k == 0:
            
            # формула 21 при k==0
            
            xAk1 = np.zeros((n*(s+1),1))
            xAk1Temp = np.dot(F,x0)+np.dot(psi,uk)
            
            for i in range(n):
                xAk1[i][0] = xAk1Temp[i][0]
            
            for i in range(s):
                xAk1Temp = np.dot(dF[i],x0)+np.dot(F,dx0[i])+np.dot(dPsi[i],uk)
                for j in range(n):
                    xAk1[i*n+n+j][0] = xAk1Temp[j][0]
        else:
            
            # ----- шаг 5
            
            # формула 26
            KTk = np.dot(F, Kk)
            
            dKTk = [ np.zeros((n,m)) for _ in range(s)]
            
            for i in range(s):
                dKTk[i] = np.dot(dF[i],Kk)+ np.dot(F, dKk[i])
            
            # ----- шаг 6
            
            # формула 23
            Fatk = np.zeros((n*(s+1),n*(s+1)))
            
            for i in range(n):
                for j in range(n):
                    Fatk[i][j] = F[i][j]
            
            # первый столбец
            for step in range(s):
                FatkTemp = dF[step] - np.dot(KTk, dH[step])
                for i in range(n):
                    for j in range(n):
                        Fatk[step*n+n+i][j] = FatkTemp[i][j]
            
            # диагонали
            FatkTemp = F - np.dot(KTk,H)
            for step in range(s):
                for i in range(n):
                    for j in range(n):
                        Fatk[step*n+n+i][step*n+n+j] = FatkTemp[i][j]
            
            
            # формула 21 при k!=0
            xAk1 = np.dot(Fatk, xA) + np.dot(psiA, uk)
        
        # ----- шаг 8

        # формула 10 
        Pk1k = ABAt(F,Pkk) + ABAt(G,Q)
        # формула 12 (B на шаге k+1)
        Bk1 = ABAt(H,Pk1k) + R
        # формула 13 (K на шаге k+1)
        Kk1 = np.dot(np.dot(Pk1k, H.transpose()), np.linalg.inv(Bk1))
        # формула 15 
        Pk1k1 = np.dot((np.eye(n)-np.dot(Kk1,H)),Pk1k)
        
        
        psiAdu = np.zeros((n*(s+1),1))
        
        # ----- шаг 9
        for b in range(N):
            
            # ----- шаг 10-12
            
            for a in range(r):
                
                if b != k:
                    psiAdu = np.zeros((n*(s+1),r))
                else:
                    dua = np.zeros((r,1))
                    dua[a][0] = 1
                    psiAdu = np.dot(psiA, dua)
                
                
                if k == 0:
                    dxAk[a][k][b] = psiAdu.copy()
                else:
                    dxAk[a][k][b] = np.dot(Fatk, dxAk[a][k-1][b]) + psiAdu
                
            
            # ----- шаг 13            
            
            for a in range(r):
            
                #fl = np.expand_dims(dxAk[a][k][b], axis=0).transpose()
                h1 = np.dot(dxAk[a][k][b],xAk1.transpose())+np.dot(xAk1,dxAk[a][k][b].transpose())
                
                c0 = cI(0, s, n)
                c0T = c0.T
                invB = np.linalg.inv(Bk1)
                
                # формула 20
                for i in range(s):
                    for j in range(s):
                        
                        h2 = np.dot(dH[i], c0)
                        h2 = np.dot(h2,h1)
                        h2 = np.dot(h2,c0T)
                        h2 = np.dot(h2,dH[j].T)
                        h2 = np.dot(h2, invB)
                        
                        h3 = np.dot(dH[i], c0)
                        h3 = np.dot(h3,h1)
                        h3 = np.dot(h3,(cI(j+1, s, n)).T)
                        h3 = np.dot(h3,H.T)
                        h3 = np.dot(h3, invB)
                        
                        h4 = np.dot(H, cI(i+1, s, n))
                        h4 = np.dot(h4,h1)
                        h4 = np.dot(h4,c0T)
                        h4 = np.dot(h4,dH[j].T)
                        h4 = np.dot(h4, invB)
                        
                        h5 = np.dot(H, cI(i+1, s, n))
                        h5 = np.dot(h5,h1)
                        h5 = np.dot(h5,(cI(j+1, s, n)).T)
                        h5 = np.dot(h5,H.T)
                        h5 = np.dot(h5, invB)
                        
                        dMtheta[a][b][i][j] += h2[0][0] + h3[0][0] + h4[0][0] + h5[0][0]
            
        Kk = Kk1.copy()
        Pkk = Pk1k1.copy()
        xA = xAk1.copy()
    
    return(dMtheta)

# ----------------------------------------------------------------------------
# X[M(e)] для критерия A-оптимальности
def XMA(M):
    return np.trace(np.linalg.inv(M))

# ----------------------------------------------------------------------------
# ИМФ точек плана и всего плана
def MiM(U, pi, s, q):
    
    # ИМФ точек плана
    Mi = [ np.zeros((s,s)) for _ in range(q)]
    for i in range(q):
        Mi[i] = IMF(U[i])
    
    # формула 38 - информационная матрица плана
    M = np.zeros((s,s))
    for i in range(q):
        M += Mi[i]*pi[i]
    
    return Mi, M

# ----------------------------------------------------------------------------
# задаём вектор с u определённый вид, необходимый для других функций
def Uformat(U1, q, N, r):
    
    U = []
    for i in range(q):
        U_help = []
        for j in range(N):
            U_help.append(np.array(U1[(i*r*N+j*r):(i*r*N+j*r+r)]))
        U.append(U_help)
    
    return U

# ----------------------------------------------------------------------------
# создаём новый план с новой точкой
def newEps(U, Uk, p, tau):
    
    Ures =  U.copy()
    pi = p.copy()
    
    q = len(pi)
    
    for i in range(q):
        pi[i] = pi[i]*(1-tau)
    
    pi = np.append(pi, tau[0])
    
    for i in range(Uk.shape[0]):
        Ures = np.append(Ures, Uk[i])
    
    return Ures, pi, q+1

# ----------------------------------------------------------------------------
# X[M(e)]-->min
def XMe(tau, params):
    
    Ustart = params['U'].copy()
    Uk = params['Uk']
    pi = params['pi'].copy()
    r = params['r']
    N = params['N']
    s = params['s']
    
    Ustart, pi, q = newEps(Ustart, Uk, pi, tau)
    
    U = Uformat(Ustart, q, N, r)
    
    # ИМФ точек плана
    Mi = [ np.zeros((s,s)) for _ in range(q)]
    for i in range(q):
        Mi[i] = IMF(U[i])
    
    # формула 38 - информационная матрица плана
    M = np.zeros((s,s)) 
    for i in range(q):
        M += Mi[i]*pi[i]
        
    return XMA(M)

# ----------------------------------------------------------------------------
# производная параметра m
def dmA(Uk, params):
    
    M = params['M']
    r = params['r']
    N = params['N']
    
    U = Uformat(Uk, 1, N, r)
    
    Minv = np.linalg.inv(M)
    
    # производная ИМФ точек плана
    Mti = dIMF(U[0])
    
    # градиент по u
    dXu = []
    
    for b in range(N):
        for a in range(r):
            dXu.append(-np.trace(np.dot(np.dot(Minv, Minv), Mti[a][b])))
            
    return dXu

# ----------------------------------------------------------------------------
# очистка плана
def cleanEps(U, pi, N, r):
    
    q = len(pi)
    delta = 0.001

    # -- сортируем точки плана по весам
    uSort = []
    piSort = pi.copy()
    numberSort = [i for i in range(q)]
    
    for i in range(q-1):
        for j in range(q-i-1):
            if pi[j] <= pi[j+1]:
                numberSort[j], numberSort[j+1] = numberSort[j+1], numberSort[j]
                piSort[j], piSort[j+1] = piSort[j+1], piSort[j]
    
    for i in range(q):
        uSort.append(U[numberSort[i]])
    
    
    pointsToDel = []
    # -- ищем точки, тяготеющие к одной группе
    for i in range(q-1):
        
        if i not in pointsToDel:
            
            uTake = np.array(uSort[i])
            uSame = []
            
            for j in range(i+1, q):
                
                if j not in pointsToDel:
                    
                    check = uTake - np.array(uSort[j])
                    check = np.dot(check.transpose(), check)
                    
                    if check <= delta:
                        uSame.append(j)
                        pointsToDel.append(j)
            
            for j in uSame:
                piSort[i] = piSort[i] + piSort[j]
    
    # -- ищем точки, с близкими к нулю весами, не тяготеющие к группам
    for i in range(q):
        if i not in pointsToDel:
            if piSort[i] < delta:
                pointsToDel.append(i)
    
    # -- очищаем план
    resU = []
    resP = []
    for i in range(q):
        if i not in pointsToDel:
            resP.append(piSort[i])
            for b in range(N):
                for a in range(r):
                    resU.append(uSort[i][b][a])
    
    return resU, resP, len(resP)
    

# ----------------------------------------------------------------------------
# параметр m заданной точки заданного плана
def mA(Uk, params):
    
    M = params['M']
    r = params['r']
    N = params['N']
    
    U = Uformat(Uk, 1, N, r)
    
    Mi = IMF(U[0])
    
    Minv = np.linalg.inv(M)
    
    return -np.trace(np.dot(np.dot(Minv,Minv),Mi))
    
# ----------------------------------------------------------------------------
# лабораторная 5
def main():
    
    # --- шаг 1
    
    r, N, q, s = initValues(1)
    
    # задаём начальный план
    Uk = np.random.uniform(0, 5, size=r*N*q)
    #Uk = np.ones(r*N*q)
    UStart = Uk.copy()
    pik = [ 1/q for _ in range(q)]
    piStart = pik.copy()
    delta = 1e-6
    
    # ИМФ точек плана eps(k=0) и всего плана
    Mi, M = MiM(Uformat(Uk, q, N, r), pik, s, q)
    Mstart = M.copy()
    flag2 = True
    k = -1
    
    while flag2 ==True:
        
        k = k+1        
        flag = True
        
        UkStep2 = Uk.copy()
        
        while flag == True:
            
            # --- шаг 2
            
            # ищем локальный максимум параметра m
            
            result = minimize(
                fun = mA,
                jac = dmA,
                x0 = np.random.uniform(0, 5, size=r*N),
                args={
                    'M': M,
                    'N': N,
                    'r': r,
                },
                method='SLSQP',
                bounds=Bounds([0]*N*r, [5]*N*r)
            )
            
            # проверка завершения алгоритма
            
            Minv = np.linalg.inv(M)
            n = np.trace(Minv)
            UkStep2 = result['x']
            Mi = IMF(Uformat(UkStep2, 1, N, r)[0])
            m = np.trace(np.dot(np.dot(Minv,Minv),Mi))
            
            if np.abs(m-n) <= delta:
                flag2 = False
                flag = False
            
            if m > n:
                flag = False
        
        
        if flag2 == False:
            Mi, M = MiM(Uformat(Uk, q, N, r), pik, s, q)
            break
        
        
        # --- шаг 3
        
        result = minimize(
            fun = XMe,
            x0 = np.random.uniform(0, 1),
            args={
                'U': Uk,
                'Uk': UkStep2,
                'pi': pik,
                'r': r,
                'N': N,
                's': s
            },
            method='SLSQP',
            bounds=Bounds(0, 1)
        )

        # добавляем новую точку в план
        Uk, pik, q = newEps(Uk, UkStep2, pik, result['x'])
        
        # --- шаг 4
        
        Uk, pik, q = cleanEps(Uformat(Uk, q, N, r), pik, N, r)
        
        Mi, M = MiM(Uformat(Uk, q, N, r), pik, s, q)
    
    
    print("\n____________________________________\n")
    print("Исходный план: ", XMA(Mstart))
    print("\nU")
    print(UStart)
    print("\npi")
    print(piStart)
    
    print("\nЗначение критерия полученного плана: X[M(e)] = ", XMA(M))
    print("\nU")
    print(Uk)
    print("\npi")
    print(pik)
    print("\n____________________________________\n")
    
    Uk[N*q*r-1] = 0
    print(Uk)
    Mi, M = MiM(Uformat(Uk, q, N, r), pik, s, q)
    print("\n X[M(e)] изменённой матрицы = ", XMA(M))
    

np.random.seed(5)
main()

