import warnings
from typing import Optional

from sklearn.svm import OneClassSVM

from qiskit_machine_learning.algorithms.serializable_model import SerializableModelMixin
from qiskit_machine_learning.exceptions import QiskitMachineLearningWarning
from qiskit_machine_learning.kernels import BaseKernel, FidelityQuantumKernel

from qiskit_machine_learning.utils import algorithm_globals


class OneClassQSVM(OneClassSVM, SerializableModelMixin):
    r"""Quantum One-Class Support Vector Machine (OneClassQSVM) that extends the scikit-learn
    `sklearn.svm.OneClassSVM <https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html>`_
    and introduces an additional `quantum_kernel` parameter.

    This class demonstrates how to use a quantum kernel for anomaly detection or novelty detection
    with a One-Class SVM. The class inherits methods like ``fit`` and ``predict`` from scikit-learn.
    Read more in the `scikit-learn user guide
    <https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection-with-svm>`_.

    **Example**

    .. code-block::

        ocqsvm = OneClassQSVM(quantum_kernel=qkernel)
        ocqsvm.fit(sample_train)
        ocqsvm.predict(sample_test)
    """

    def __init__(self, *, quantum_kernel: Optional[BaseKernel] = None, **kwargs):
        """
        Args:
            quantum_kernel: A quantum kernel to be used for anomaly detection.
            Must be ``None`` when a precomputed kernel is used. If None,
            defaults to :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel`.
            **kwargs: Arbitrary keyword arguments to pass to OneClassSVM constructor.
        """
        if "kernel" in kwargs:
            msg = (
                "'kernel' argument is not supported and will be discarded, "
                "please use 'quantum_kernel' instead."
            )
            warnings.warn(msg, QiskitMachineLearningWarning, stacklevel=2)
            # if we don't delete, then this value clashes with our quantum kernel
            del kwargs["kernel"]
        if quantum_kernel is None:
            msg = "No quantum kernel is provided, SamplerV1 based quantum kernel will be used."
            warnings.warn(msg, QiskitMachineLearningWarning, stacklevel=2)
        self._quantum_kernel = quantum_kernel if quantum_kernel else FidelityQuantumKernel()

        # if "random_state" not in kwargs:
        #     kwargs["random_state"] = algorithm_globals.random_seed

        super().__init__(kernel=self._quantum_kernel.evaluate, **kwargs)

    @property
    def quantum_kernel(self) -> BaseKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: BaseKernel):
        """Sets quantum kernel"""
        self._quantum_kernel = quantum_kernel
        self.kernel = self._quantum_kernel.evaluate

    # we override this method to be able to pretty print this instance
    @classmethod
    def _get_param_names(cls):
        names = OneClassSVM._get_param_names()
        names.remove("kernel")
        return sorted(names + ["quantum_kernel"])