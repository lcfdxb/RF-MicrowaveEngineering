"""
* Software License Agreement (BSD License)
*
*  Copyright (c) 2020, Chengfeng Luo.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Kentaro Wada nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
"""
# single_stub.py
# For calculating single stub tuning
# Also will plot smith chart
# 
import numpy as np
from numpy import cos,sin,sqrt,pi,arctan
from cmath import phase
from matplotlib import pyplot as plt

# Input parameters here
z0 = 50
zl = 60 - 80j


y0 = 1/z0

def draw_arc(c_x, c_y, r, start_phase, end_phase, direction='clockwise'):
    # input phase range: -pi:pi
    if direction == 'clockwise':
        if end_phase>start_phase:
            end_phase -= 2*np.pi
        theta = np.linspace(start_phase, end_phase, 1000)
    else:
        if start_phase>end_phase:
           start_phase -= 2*np.pi
        theta = np.linspace(start_phase, end_phase, 1000) 
    x = r*cos(theta)+c_x
    y = r*sin(theta)+c_y
    return x,y

def draw_circle(c_x,c_y,r):
    # input:
    # c_x: center's x
    # c_y: center's y
    # r  : radious
    # output: x,y array for plotting
    theta = np.linspace(0, 2*np.pi, 1000)
    x = r*cos(theta)+c_x
    y = r*sin(theta)+c_y
    return x,y

def z2gamma(z):
    global z0
    return (z-z0)/(z+z0)

def y2gammaZ(y):
    global z0
    y0 = 1/z0
    return (y-y0)/(y+y0)

# find t use 5.9 at p.238
rl = zl.real
xl = zl.imag
t1 = (xl+sqrt(rl*((z0-rl)**2+xl**2)/z0))/(rl-z0)
t2 = (xl-sqrt(rl*((z0-rl)**2+xl**2)/z0))/(rl-z0)
# compute d1,d2 and y1,y2
d1 = (arctan(t1)/2/pi).real
d2 = (arctan(t2)/2/pi).real
if d1<0: d1 += 1/2
if d2<0: d2 += 1/2
g1 = rl*(1+t1**2)/(rl**2+(xl+z0*t1)**2)
g2 = rl*(1+t2**2)/(rl**2+(xl+z0*t2)**2)
b1 = (rl**2*t1-(z0-xl*t1)*(xl+z0*t1))/(z0*(rl**2+(xl+z0*t1)**2))
b2 = (rl**2*t2-(z0-xl*t2)*(xl+z0*t2))/(z0*(rl**2+(xl+z0*t2)**2))
yl = 1/zl
y1 = complex(g1,b1)
y2 = complex(g2,b2)
l1open  = (-arctan(b1/y0)/2/pi).real
if l1open<0: l1open += 1/2
l1short = (arctan(y0/b1)/2/pi).real
if l1short<0: l1short += 1/2
l2open  = (-arctan(b2/y0)/2/pi).real
if l2open<0: l2open += 1/2
l2short = (arctan(y0/b2)/2/pi).real
if l2short<0: l2short += 1/2

# draw unit gamma circle, G=1 circle and axis
fig, ax = plt.subplots(1)
ax.plot([-1.1,1.1],[0,0],'grey')
ax.plot([0,0],[-1.1,1.1],'grey')
x,y = draw_circle(0,0,1)
ax.plot(x, y, 'black')
g1_x,g1_y = draw_circle(0.5,0,0.5)
ax.plot(g1_x, g1_y, 'lightpink')
ax.text(0.5-sqrt(2)/4,-sqrt(2)/4,'$G=1$', 
        horizontalalignment='right',
        verticalalignment='top')
ax.set_aspect(1)
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.xlabel(u'Re{$\Gamma$}')
plt.ylabel(u'Im{$\Gamma$}')
plt.title(u'Smith Z chat\n'+ 
        'For open circuit: $l_1$=%.3f$\lambda$  $l_2$=%.3f$\lambda$\n'%(l1open,l2open) +
        'For short circuit: $l_1$=%.3f$\lambda$  $l_2$=%.3f$\lambda$'%(l1short,l2short))
plt.grid(linestyle='--')

# draw constant SWR line from YL to Y1,Y2
zl_plt = z2gamma(zl)
yl_plt = y2gammaZ(yl)
y1_plt = y2gammaZ(y1)
y2_plt = y2gammaZ(y2)
x,y = draw_arc(0,0,abs(yl_plt),phase(yl_plt),phase(y1_plt),'clockwise')
ax.plot(x, y, ':')
ax.text(x[len(x)//2], y[len(y)//2],u'$d_1$=%.3f$\lambda$'%d1)
x,y = draw_arc(0,0,abs(yl_plt),phase(yl_plt),phase(y2_plt),'clockwise')
ax.plot(x, y, ':')
ax.text(x[len(x)//2], y[len(y)//2],u'$d_2$=%.3f$\lambda$'%d2)
# draw ZL, YL, Y1 and Y2
plt.plot(zl_plt.real,zl_plt.imag,'o',color='orangered')
plt.text(zl_plt.real,zl_plt.imag,'$Z_L$')
plt.plot(yl_plt.real,yl_plt.imag,'o',color='limegreen')
plt.text(yl_plt.real,yl_plt.imag,'$Y_L$=%.3f+j%.3f'%(yl.real/y0,yl.imag/y0))
plt.plot(y1_plt.real,y1_plt.imag,'o',color='limegreen')
plt.text(y1_plt.real,y1_plt.imag,'$Y_1$=%.3f+j%.3f'%(y1.real/y0,y1.imag/y0))
plt.plot(y2_plt.real,y2_plt.imag,'o',color='limegreen')
plt.text(y2_plt.real,y2_plt.imag,'$Y_2$=%.3f+j%.3f'%(y2.real/y0,y2.imag/y0))

print('**************************************************')
print('Single Stub Turning Result')
print('**************************************************')
print('Input:')
print('  Z0 = %.5e'%z0)
print('  ZL = %s'%zl)
print('Solution 1:')
print('  Y1 = %.5e+%.5ej'%(y1.real,y1.imag))
print('  For open circuit:')
print('    d = %.5e'%(l1open))
print('  For short circuit:')
print('    d = %.5e'%(l1short))
print('Solution 2:')
print('  Y1 = %.5e+%.5ej'%(y2.real,y2.imag))
print('  For open circuit:')
print('    d = %.5e'%(l2open))
print('  For short circuit:')
print('    d = %.5e'%(l2short))

plt.show()