import numpy as np
import matplotlib.pyplot as plt


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

plt.rcParams['figure.figsize'] = [5, 2.5]
plt.figure(layout='constrained')


mass = [451.47,415,446]
label = ['our 1D beam model','Sarojini et al.\n beam model','Sarojini et al.\n shell model']
colors = ['orangered','gray','gray']


plt.bar(label,mass,color=colors,width=0.6,edgecolor='black')
plt.ylabel('PEGASUS wing mass (kg)')


plt.text(0,418,'451 kg',ha='center')
plt.text(1,380,'415 kg',ha='center')
plt.text(2,410,'446 kg',ha='center')


plt.savefig('peg.png', dpi=800, bbox_inches='tight')


plt.show()