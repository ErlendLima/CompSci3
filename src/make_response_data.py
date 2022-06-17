import ompy as om
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

head = Path(om.__file__).parent.parent
oscar2017 = Path(head / "OCL_response_functions/oscar2017_scale1.15/")
oscar2020 = Path(head / "OCL_response_functions/oscar2020/mama_export/")

#R, _, _ = om.Response.LoadDir(oscar2020)
R = om.Response(oscar2017)
R.get_probabilities()

print(R.pFE)

data = {'Eg': R.resp.Eg,
        'FE': R.pFE,
        'SE': R.pSE,
        'DE': R.pDE,
        '511': R.p511}
df = pd.DataFrame(data)
#R = R.rename({'FWHM_rel_norm': "FWHM"}, axis='columns')
df.plot(x='Eg', y='SE', kind='scatter')

plt.show()
df.to_csv('response_p2.csv')
