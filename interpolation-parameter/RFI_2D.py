import numpy as np
import matplotlib.pyplot as plt
#import rfinterpolation as rfi
import sys



LORES = 15
HIRES = 30



def getNearest(xArr, xVal):
    l = 0
    r = len(xArr) - 1
    while l < r:
        mid = int((l + r)/2)
        if xArr[mid] < xVal:
            l = mid + 1
        elif xArr[mid] > xVal:
            r = mid - 1
        # If a value is exactly matched
        elif xArr[mid] == xVal:
            if mid > 0:
                l = mid
                r = mid - 1
            else:
                l = mid + 1
                r = mid

            break

    if l == r:
        if xArr[r] > xVal and r > 0:
            r -= 1
        elif xArr[l] < xVal and l < len(xArr) - 1:
            l += 1
        # If a value is exactly matched
        elif r > 0:
            r -= 1
        elif l < len(xArr) - 1:
            l += 1

    if l <= 0:
        exit("l is too smol")
    if r >= len(xArr):
        exit("r is too big")

    return r, xArr[r], xArr[l]

            

def RFI_1D_interval(x0s, i, z0s, x1):
    N = len(x0s)

    delta = x0s[1:] - x0s[:-1]
    S = (z0s[1:] - z0s[:-1])*np.power(delta, -1)    

    if i == 0:
        C2 = (S[1] - S[0])/(delta[1] + delta[0])
        if S[0]*(S[0] - delta[0]*C2) <= 0:
            C2 = S[0]/delta[0]
        z1 = z0s[0] + (x1 - x0s[0])*(S[0] - C2*(x0s[1] - x1))

    elif i == N - 2:
        C1 = (S[N-2] - S[N-3])/(delta[N-2] + delta[N-3])
        z1 = z0s[-2] + (x1 - x0s[-2])*(S[-2] - C1*(x0s[-1] - x1))

    else:
        C1 = (S[i] - S[i-1])/(delta[i] + delta[i-1])
        if i == 1 and S[i-1]*(S[i-1] - delta[i-1]*C1) <= 0:
            C1 = (S[i] - 2*S[i-1])/delta[i]
            
        C2 = (S[i+1] - S[i])/(delta[i+1] + delta[i])
        
        mu1 = np.abs(C2*(x0s[i+1] - x1))
        mu2 = np.abs(C1*(x1 - x0s[i]))
        if mu1 > sys.float_info.epsilon and mu2 > sys.float_info.epsilon:
            z1 = (z0s[i] 
                  + (x1 - x0s[i])
                  *( S[i] 
                     - (C1*mu1 + C2*mu2)
                     /(mu1 + mu2)
                     *(x0s[i+1] - x1) ))
        else:
            z1 = z0s[i] + (x1 - x0s[i])*S[i] 

    return z1



def RFI_1D(x0s, z0s, x1s):
    N = len(x0s)
    
    z1s = np.zeros(np.asarray(x1s).shape)

    for i0 in range(N-1):
        # i1 represents the index of x1s that we are evaluating
        i1, xLo, xHi = getNearest(x1s, x0s[i0])
        if x1s[i1] < x0s[i0]:
            i1 += 1

        while x1s[i1] <= x0s[i0+1]:
            z1s[i1] = RFI_1D_interval(x0s, i0, z0s,   x1s[i1])
            if i1 < len(x1s) - 1:
                i1 += 1
            else:
                break

    return z1s



# The x1s and y1s arrays represent the "x" and "y" variables from Kerley 1977
def RFI_2D(x0s, y0s, x1s, y1s, zz0s):
    zz1s = np.zeros((len(y1s), len(x1s)))

    N = len(x0s)
    M = len(y0s)
    
    for i0 in range(N-1):
        for j0 in range(M-1):
            i1, xLo, xHi = getNearest(x1s, x0s[i0])
            if x1s[i1] < x0s[i0]:
                i1 += 1

            # Iterate through i1 until x is not between x_i and x_i+1
            while x1s[i1] <= x0s[i0+1]:
                j1, yLo, yHi = getNearest(y1s, y0s[j0])
                if y1s[j1] < y0s[j0]:
                    j1 += 1

                # Iterate through j1 until y is not between y_i and y_i+1
                while y1s[j1] <= y0s[j0+1]:
                    # These represent four different 1D interpolants in 
                    # the domain:
                    # zi0a -> z[j0, i0]   to z[j0, i0+1]
                    # zi0b -> z[j0+1, i0] to z[j0+1, i0+1]
                    # zj0a -> z[j0, i0]   to z[j0+1, i0]
                    # zj0b -> z[j0, i0+1] to z[j0+1, i0+1]
                    zi0a = RFI_1D_interval(x0s, i0, zz0s[j0],   x1s[i1])
                    zi0b = RFI_1D_interval(x0s, i0, zz0s[j0+1], x1s[i1])
                    zj0a = RFI_1D_interval(y0s, j0, zz0s[:, i0],   y1s[j1])
                    zj0b = RFI_1D_interval(y0s, j0, zz0s[:, i0+1], y1s[j1])

                    qx = (x1s[i1] - x0s[i0])/(x0s[i0+1] - x0s[i0])
                    qy = (y1s[j1] - y0s[j0])/(y0s[j0+1] - y0s[j0])
                    
                    zz1s[j1, i1] = (zi0a*(1 - qy) 
                                    + zi0b*qy
                                    + zj0a*(1 - qx)
                                    + zj0b*qx
                                    - zz0s[j0,   i0  ]*(1 - qx)*(1 - qy)
                                    - zz0s[j0+1, i0  ]*(1 - qx)*qy
                                    - zz0s[j0,   i0+1]*qx*(1 - qy)
                                    - zz0s[j0+1, i0+1]*qx*qy)
                    j1 += 1
                    if j1 >= len(y1s):
                        break

                i1 += 1
                if i1 >= len(x1s):
                    break


    return zz1s



def example2D_1():
    fig, axs = plt.subplots(2, figsize=(4, 8))

    x0 = np.linspace(0, np.pi*2, LORES)
    y0 = np.linspace(0, np.pi*2, LORES)
    x1 = np.linspace(0, np.pi*2, HIRES)
    y1 = np.linspace(0, np.pi*2, HIRES)

    xx0, yy0 = np.meshgrid(x0, y0)
    xx1, yy1 = np.meshgrid(x1, y1)
    zz0 = np.sin(xx0)*np.sin(yy0)
    zz1 = RFI_2D(x0, y0, x1, y1, zz0)

    ctf0 = axs[0].contourf(xx0, yy0, zz0)
    ctf1 = axs[1].contourf(xx1, yy1, zz1)
    plt.show()

def example1D_1():
    x0 = np.linspace(0, 2*np.pi, LORES)
    x1 = np.linspace(0, 2*np.pi, HIRES)

    z1_approx = RFI_1D(x0, np.sin(x0), x1)
    z1_approx_alt = RFI_1D_alt(x0, np.sin(x0), x1)
    z1 = np.sin(x1)

    plt.plot(x0, np.sin(x0), label="Initial")
    # plt.plot(x1, z1_approx, label="Approx")
    plt.plot(x1, z1_approx_alt, label="Approx alt")
    plt.plot(x1, z1, label="Exact", ls='--')
    plt.legend()
    plt.show()



def example1D_2():
    x0 = np.linspace(0, 1, LORES)
    x1 = np.linspace(0, 1, HIRES)

    z0 = 0.5*x0
    z0[LORES/2:] = x0[LORES/2:]
    z1_approx = RFI_1D(x0, z0, x1)
    z1 = 0.5*x1
    z1[HIRES/2:] = x1[HIRES/2:]

    plt.plot(x0, z0, label="Initial")
    plt.plot(x1, z1_approx, label="Approx")
    plt.plot(x1, z1, label="Exact")
    plt.legend()
    plt.show()



#if __name__ == "__main__":
#    # example1D_1()
#    example2D_1()
