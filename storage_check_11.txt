# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:55:25 2019

@author: Michael Stöhr, Patrik Schönfeldt
"""

import oemof.solph as solph
import oemof.outputlib as outputlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter

def flexibility(epc):

    date_time_index = pd.date_range('1/1/2018 1:00:00', periods=10, freq='H')
    energysystem = solph.EnergySystem(timeindex=date_time_index)

    bel = solph.Bus(label = 'bel')
    energysystem.add(bel)

    g = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    d = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])

    icf = 0.95   # charging efficiency
    lr = 0.1     # loss rate of energy in the storage
    ocf = 0.95   # discharging efficiency
    c = 1        # cost of electricity
    
    energysystem.add(solph.Source(label='generation',
            outputs={bel: solph.Flow(actual_value=g,
                    fixed=True, nominal_value=1,
                    variable_costs = 0)}))

    energysystem.add(solph.Sink(label='demand',
            inputs={bel: solph.Flow(actual_value=d,
                    fixed=True, nominal_value=1,
                    variable_costs = 0)}))

    energysystem.add(solph.Sink(label='curtailment',
            inputs={bel: solph.Flow(variable_costs = 0)}))

    energysystem.add(solph.Source(label='backup',
            outputs={bel: solph.Flow(variable_costs = c)}))

    energysystem.add(solph.components.GenericStorage(
            label='storage',
            inputs={bel: solph.Flow(variable_costs = 0)},
            outputs={bel: solph.Flow(variable_costs = 0)},
            initial_storage_level = None,
            balanced = True,
            loss_rate = lr,
            inflow_conversion_factor = icf,
            outflow_conversion_factor = ocf,
            min_storage_level = 0,
            max_storage_level = 1,
            investment=solph.Investment(ep_costs = epc)
            ))

    om = solph.Model(energysystem)
    om.solve(solver='cbc', solve_kwargs={'tee': True})

    results = outputlib.processing.results(om)

    node_backup = energysystem.groups['backup']
    node_storage = energysystem.groups['storage']

    df = pd.DataFrame
    df = results[(node_backup, bel)]['sequences']
    df = df.rename(index=str, columns={'flow': 'backup'})

    backup_energy = df['backup'].sum()
    backup_costs = backup_energy*c
    storage_nominal_capacity = results[node_storage, None]['scalars']['invest']
    storage_investment_costs = storage_nominal_capacity * epc
    costs_of_flexibility = backup_costs + storage_investment_costs
    cntr = 3 - backup_energy - epc/c * storage_nominal_capacity
   
    r = [backup_energy, storage_nominal_capacity, backup_costs, 
         storage_investment_costs, costs_of_flexibility, cntr]
    
    return r

###############################################################################


epc = 0     # costs per unit storage capacity for all time steps together
epc = []
backup_energy = []
storage_nominal_capacity = []
backup_costs = []
storage_investment_costs = []
costs_of_flexibility = []
check = []

beg = 0
end = 1.2
int = 200
step = (end-beg)/int
rg = np.arange(beg, end+step, step)

for i in rg:
    epc.append(i)
    r = flexibility(i)
    backup_energy.append(r[0])
    storage_nominal_capacity.append(r[1])
    backup_costs.append(r[2])
    storage_investment_costs.append(r[3])
    costs_of_flexibility.append(r[4])
    check.append(r[5])
    
X = rg
F1 = costs_of_flexibility
F2 = backup_costs
F3 = storage_investment_costs
F4 = backup_energy
F5 = storage_nominal_capacity
F6 = check

fig, ax = plt.subplots()
plt.plot(X,F1,'k-', label='costs of flexibility')
plt.plot(X,F2,'k--', label='backup costs')
plt.plot(X,F3,'k:', label='storage investment costs')
ax.set(xlabel='epc', ylabel='costs', title='costs of flexibility')
legend = ax.legend(loc='best', bbox_to_anchor=(0.01, 0., 0.5, 1.), shadow=True, fontsize=10)
legend.get_frame().set_facecolor('C0')
fig.savefig("costs of flexibility.png")
plt.show()

fig, ax = plt.subplots()
plt.plot(X,F4,'k-', label='backup energy')
plt.plot(X,F5,'k--', label='storage nominal capacity')
plt.plot(X,F6,'k:', label='check')
ax.set(xlabel='epc', ylabel='energy', title='storage vs backup')
legend = ax.legend(loc='best', bbox_to_anchor=(0.01, 0., 0.5, 0.5), shadow=True, fontsize=10)
legend.get_frame().set_facecolor('C0')
fig.savefig("storage vs backup.png")
plt.show()

dict_r = {'epc': epc, 'backup_energy': backup_energy,
          'storage_nominal_capacity': storage_nominal_capacity,
          'backup_costs': backup_costs,
          'storage_investment_costs': storage_investment_costs,
          'costs_of_flexibility': costs_of_flexibility,
          'check': check}

df_r = pd.DataFrame(dict_r, 
            columns=['epc',
                     'backup_energy', 
                     'storage_nominal_capacity',
                     'backup_costs',
                     'storage_investment_costs',
                     'costs_of_flexibility', 
                     'check'])

writer = ExcelWriter('basic_flexibility.xlsx')
df_r.to_excel(writer,'Sheet1')
writer.save()

