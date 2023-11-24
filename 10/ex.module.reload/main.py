import module
#from imp import reload
from importlib import reload

module.p()
module.a = 20
module.p()
module = reload(module)
module.p()

