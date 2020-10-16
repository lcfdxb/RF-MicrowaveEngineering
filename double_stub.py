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
from numpy import cos, sin, tan, sqrt, pi, arctan
from cmath import phase
from matplotlib import pyplot as plt
import sys

# Input parameters here
z0 = 50
zl = 60 - 80j
d = 1/8        # unit: lambda=1

y0 = 1/z0
yl = 1/zl


def draw_arc(c_x, c_y, r, start_phase, end_phase, direction='clockwise'):
    # input phase range: -pi:pi
    if direction == 'clockwise':
        if end_phase > start_phase:
            end_phase -= 2*np.pi
        theta = np.linspace(start_phase, end_phase, 1000)
    else:
        if start_phase > end_phase:
           start_phase -= 2*np.pi
        theta = np.linspace(start_phase, end_phase, 1000)
    x = r*cos(theta)+c_x
    y = r*sin(theta)+c_y
    return x, y

def draw_arc_shortest(c_x, c_y, r, start_phase, end_phase):
    # input phase range: -pi:pi
    diff = abs(start_phase-end_phase)
    if diff < pi:
        theta = np.linspace(start_phase, end_phase, 1000)
    else:
        if start_phase < end_phase:
            start_phase += 2*np.pi
        else: 
            end_phase += 2*np.pi
        theta = np.linspace(start_phase, end_phase, 1000)
    x = r*cos(theta)+c_x
    y = r*sin(theta)+c_y
    return x, y


def draw_circle(c_x, c_y, r):
    # input:
    # c_x: center's x
    # c_y: center's y
    # r  : radious
    # output: x,y array for plotting
    theta = np.linspace(0, 2*np.pi, 1000)
    x = r*cos(theta)+c_x
    y = r*sin(theta)+c_y
    return x, y


def z2gamma(z):
    global z0
    return (z-z0)/(z+z0)


def y2gammaZ(y):
    global z0
    y0 = 1/z0
    return (y-y0)/(y+y0)


# follow textbook p.245
rl = zl.real
xl = zl.imag
gl = yl.real
bl = yl.imag
t = tan(2*pi*d)

# check if feasible
if gl > y0*(1+t**2)/(t**2):
    print("No solution!")
    sys.exit(1)

# solution 1: y11, y21
# solution 2: y12, y22
b11 = (y0+sqrt((1+t**2)*gl*y0-gl**2*(t**2)))/t - bl
b12 = (y0-sqrt((1+t**2)*gl*y0-gl**2*(t**2)))/t - bl
b21 = (+y0*sqrt(y0*gl*(1+t**2)-gl**2*(t**2))+gl*y0)/gl/t
b22 = (-y0*sqrt(y0*gl*(1+t**2)-gl**2*(t**2))+gl*y0)/gl/t

y11 = gl + (bl+b11)*1j
y12 = gl + (bl+b12)*1j
y21 = y0*(gl+1j*(bl+b11+y0*t))/(y0+1j*t*(gl+1j*(bl+b11)))
y22 = y0*(gl+1j*(bl+b12+y0*t))/(y0+1j*t*(gl+1j*(bl+b12)))

l11open  = (arctan(b11/y0)/2/pi)
if l11open<0: l11open += 1/2
l11short = (-arctan(y0/b11)/2/pi)
if l11short<0: l11short += 1/2
l21open  = (arctan(b21/y0)/2/pi)
if l21open<0: l21open += 1/2
l21short = (-arctan(y0/b21)/2/pi)
if l21short<0: l21short += 1/2

l12open  = (arctan(b12/y0)/2/pi)
if l12open<0: l12open += 1/2
l12short = (-arctan(y0/b12)/2/pi)
if l12short<0: l12short += 1/2
l22open  = (arctan(b22/y0)/2/pi)
if l22open<0: l22open += 1/2
l22short = (-arctan(y0/b22)/2/pi)
if l22short<0: l22short += 1/2

zl_plt = z2gamma(zl)
yl_plt = y2gammaZ(yl)
y11_plt = y2gammaZ(y11)
y12_plt = y2gammaZ(y12)
y21_plt = y2gammaZ(y21)
y22_plt = y2gammaZ(y22)

# draw unit gamma circle, G=1 circle and axis
fig, ax=plt.subplots(1)
ax.plot([-1.1, 1.1], [0, 0], 'grey')
ax.plot([0, 0], [-1.1, 1.1], 'grey')
x, y=draw_circle(0, 0, 1)
ax.plot(x, y, 'black')
g1_x, g1_y=draw_circle(0.5, 0, 0.5)
ax.plot(g1_x, g1_y, color = 'lightpink')
ax.text(0.5-sqrt(2)/4, -sqrt(2)/4, '$G=1$',
        horizontalalignment='right',
        verticalalignment='top')
g1_x, g1_y=draw_circle(0.5*cos(4*pi*d), 0.5*sin(4*pi*d), 0.5)
ax.plot(g1_x, g1_y, color = 'lightpink')
ax.text(g1_x[len(g1_x)//2], g1_y[len(g1_x)//2], '$G=1$ at\nd=%.3f$\lambda$' % d,
        horizontalalignment='right',
        verticalalignment='top')
ax.set_aspect(1)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel(u'Re{$\Gamma$}')
plt.ylabel(u'Im{$\Gamma$}')
plt.title(u'Smith Z chat\n'
          )
plt.grid(linestyle='--')

# draw constant g line for YL
gl_n = gl/y0
x,y = draw_arc_shortest(gl_n/(1+gl_n),0,1/(1+gl_n),
                        phase(yl_plt-gl_n/(1+gl_n)),
                        phase(y11_plt-gl_n/(1+gl_n)))
ax.plot(x, y, color = 'yellowgreen')
ax.text(x[len(x)//2], y[len(y)//2],u'$B_{11}$=%.3f'%(b11/y0))
x,y = draw_arc_shortest(gl_n/(1+gl_n),0,1/(1+gl_n),
                        phase(yl_plt-gl_n/(1+gl_n)),
                        phase(y12_plt-gl_n/(1+gl_n)))
ax.plot(x, y, color = 'lightblue')
ax.text(x[len(x)//2], y[len(y)//2],u'$B_{12}$=%.3f'%(b12/y0))

# draw constant SWR line
x,y = draw_arc(0,0,abs(y11_plt),phase(y11_plt),phase(y21_plt),'clockwise')
ax.plot(x, y, ':', color = 'limegreen')
x,y = draw_arc(0,0,abs(y12_plt),phase(y12_plt),phase(y22_plt),'clockwise')
ax.plot(x, y, ':', color = 'lightskyblue')

# draw constant g line for Y2
x,y = draw_arc_shortest(0.5,0,0.5,
                        phase(y21_plt-0.5),
                        phase(-0.5))
ax.plot(x, y, color = 'yellowgreen')
ax.text(x[len(x)//2], y[len(y)//2],u'$B_{21}$=%.3f'%(b21/y0),
        horizontalalignment='right',
        verticalalignment='top')
x,y = draw_arc_shortest(0.5,0,0.5,
                        phase(y22_plt-0.5),
                        phase(-0.5))
ax.plot(x, y, color = 'lightblue')
ax.text(x[len(x)//2], y[len(y)//2],u'$B_{22}$=%.3f'%(b22/y0),
        horizontalalignment='right',
        verticalalignment='top')

# draw ZL, YL, Y1 and Y2
# print normalized Z, Y for better display
plt.plot(zl_plt.real,zl_plt.imag,'o',color='orangered')
plt.text(zl_plt.real,zl_plt.imag,'$Z_L$')
plt.plot(yl_plt.real,yl_plt.imag,'o',color='cornflowerblue')
plt.text(yl_plt.real,yl_plt.imag,'$Y_L$=%.3f+j%.3f'%(yl.real/y0,yl.imag/y0)) 
plt.plot(y11_plt.real,y11_plt.imag,'o',color='olivedrab')
plt.text(y11_plt.real,y11_plt.imag,'$Y_{11}$=%.3f+j%.3f'%(y11.real/y0,y11.imag/y0))
plt.plot(y21_plt.real,y21_plt.imag,'o',color='olivedrab')
plt.text(y21_plt.real,y21_plt.imag,'$Y_{21}$=%.3f+j%.3f'%(y21.real/y0,y21.imag/y0))
plt.plot(y12_plt.real,y12_plt.imag,'o',color='steelblue')
plt.text(y12_plt.real,y12_plt.imag,'$Y_{12}$=%.3f+j%.3f'%(y12.real/y0,y12.imag/y0))
plt.plot(y22_plt.real,y22_plt.imag,'o',color='steelblue')
plt.text(y22_plt.real,y22_plt.imag,'$Y_{22}$=%.3f+j%.3f'%(y22.real/y0,y22.imag/y0))

plt.title(u'Smith Z chat\n'+ 
        'For open circuit:'+
        '  $l_{11}$=%.3f$\lambda$  $l_{21}$=%.3f$\lambda$ or'%(l11open,l21open) +
        '  $l_{12}$=%.3f$\lambda$  $l_{22}$=%.3f$\lambda$\n'%(l12open,l22open) +
        'For short circuit:'+
        '  $l_{11}$=%.3f$\lambda$  $l_{21}$=%.3f$\lambda$ or'%(l11short,l21short) +
        '  $l_{12}$=%.3f$\lambda$  $l_{22}$=%.3f$\lambda$'%(l12short,l22short) )

print('**************************************************')
print('Double Stub Turning Result')
print('**************************************************')
print('Input:')
print('  Z0 = %.5e'%z0)
print('  ZL = %s'%zl)
print('  d  = %.5e lambda'%d)
print('Solution 1:')
print('  Y11 = %.5e+%.5ej, Y21 = %.5e+%.5ej'%(y11.real,y11.imag,y21.real,y21.imag))
print('  B11 = %.5e, B21 = %.5e'%(b11,b21))
print('  For open circuit:')
print('    l1 = %.5e, l2 = %.5e'%(l11open,l21open))
print('  For short circuit:')
print('    l1 = %.5e, l2 = %.5e'%(l11short,l21short))
print('Solution 2:')
print('  Y12 = %.5e+%.5ej, Y22 = %.5e+%.5ej'%(y12.real,y12.imag,y22.real,y22.imag))
print('  B12 = %.5e, B22 = %.5e'%(b12,b22))
print('  For open circuit:')
print('    l1 = %.5e, l2 = %.5e'%(l12open,l22open))
print('  For short circuit:')
print('    l1 = %.5e, l2 = %.5e'%(l12short,l22short))

plt.show()
