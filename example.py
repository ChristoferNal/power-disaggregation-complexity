"""Example usage"""
from nilmtk import DataSet
from nilmtk.elecmeter import ElecMeterID
import nilmtk_complexity
import tracebase_complexity


# On REDD dataset using NILMTK API
print("ON REDD")
ds = DataSet('../../Datasets/REDD/redd.h5') # Path to REDD .h5 dataset
elec = ds.buildings[2].elec
meterkeys = [4, 6, 9, 10]
mgroup = elec.from_list([ElecMeterID(elec[m].instance(),elec.building(),elec.dataset()) for m in meterkeys])
print(nilmtk_complexity.compute(mgroup))

# On Tracebase dataset
print()
print("ON TRACEBASE")
pathlist = ['../../Datasets/Tracebase/Complete/Coffeemaker',
        '../../Datasets/Tracebase/Complete/Refrigerator',
        '../../Datasets/Tracebase/Complete/TV-LCD',
        '../../Datasets/Tracebase/Complete/PC-Desktop']

print(tracebase_complexity.compute(pathlist))
