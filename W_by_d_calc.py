import numpy as np
input_para = False

if input_para :
  z0 = input('Z0 = ')
  d  = input('board thickness d = ')
  eps = input('relative permittivity = ')
else:
  z0 = 100
  d  = 0.159
  eps = 2.2

# formula 1
A = z0/60*np.sqrt((eps+1)/2)+(eps-1)/(eps+1)*(0.23+0.11/eps)
res1 = 8*np.exp(A)/(np.exp(2*A)-2)
# formula 2
B = 377*np.pi/(2*z0*np.sqrt(eps))
res2 = 2/np.pi*(B-1-np.log(2*B-1)+(eps-1)/(2*eps)*
                  (np.log(B-1)+0.39-0.61/eps)
               )
print("*****************************")
print("***********Start*************")
print('Result 1 (for W/d < 2)')
print('  W/d = %s'%res1)
print('  W = %s'%(res1*d))
print('Result 2 (for W/d > 2)')
print('  W/d = %s'%res2)
print('  W = %s'%(res2*d))
print("-----------------------")
print("**Recompute Z0**")
w_by_d = res2
Epsilon_e = (eps+1)/2+(eps-1)/2/np.sqrt(1+12/w_by_d)
print('Epsilon_e = %s'%Epsilon_e)
z0_r1 = 60/np.sqrt(Epsilon_e)*np.log(8/w_by_d+w_by_d/4)
z0_r2 = 120*np.pi/(np.sqrt(Epsilon_e) * 
                   (w_by_d+1.393+0.667*np.log(w_by_d+1.444)))
print('Result 1 (W/d <= 1)')
print('  Z0 = %s'%z0_r1)
print('Result 2 (W/d >= 1)')
print('  Z0 = %s'%z0_r2)
