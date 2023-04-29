from scalib import _scalib_ext
from scalib.config import get_config

class Information:
    def __init__(self,model,max_popped_classes):
        self._inner = _scalib_ext.ItEstimator(model,max_popped_classes)

    def fit_u(self,traces,labels):
        r"""TODO
        assumes uniform leakage (for now)
        Returns : mean information, var information

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        labels : array_like, uint64
            Label for each trace. Must be of shape `(n,)` and
            must be `uint64`.
        """
        assert traces.shape[0]==labels.shape[0]
        return self._inner.fit_u(traces,labels,get_config())

    def get_information(self):
        return self._inner.get_information()