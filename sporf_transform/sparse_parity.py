import numpy as np
   
def sparse_parity(n, p_informative, p):
   
    ny0 = ny1 = int(n/2)

    pin = p_informative

    def genZ(ex = 3):
        Z = np.random.uniform(low=-1,high=1,size=n*ex*pin).reshape(n*ex, pin)

        Y = np.zeros((n*ex))
        for i in range(Z.shape[0]):
            Y[i] = np.sum(Z[i,:] > 0) % 2

        return(Z, Y)


    Z, Y = genZ(ex=1)

    ind0 = np.where(Y == 0)[0]
    ind1 = np.where(Y == 1)[0]

    j = 0
    while len(ind0) < int(n/2) or len(ind1) < int(n/2):
        Z, Y = genZ(3 + j)

        ind0 = np.where(Y == 0)[0]
        ind1 = np.where(Y == 1)[0]
        j += 1


    ind0 = ind0[0:(int(n/2))]
    ind1 = ind1[0:(int(n/2))]

    a = []

    for i in range(len(ind0)):
        a.append(ind0[i])
        a.append(ind1[i])

    Xin = Z[a, :]
    Xnoise = np.random.uniform(low=-1,high=1,size=n*(p-pin)).reshape((n,p-pin))

    X = np.hstack((Xin, Xnoise))
    Y = [0,1] * int(n/2)

    return(X, Y)

