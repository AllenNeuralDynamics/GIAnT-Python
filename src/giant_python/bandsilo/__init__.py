"""Self-contained BandSILo (SLAP2 integration/band-scan) backend.

This package owns everything the integration source-extraction backend needs
(HDF5 IO, geometry/PSF, trial-data reading, background/rho, peak detection,
NMF localization, trace extraction, and orchestration), ported faithfully from
``extractSLAP2IntegrationSources.py`` in ophys-slap2-analysis.

It is deliberately self-contained: shared-looking kernels are duplicated here
rather than promoted to the generic ``math``/``io`` homes, so a future standard
(pixel-movie) SILo backend can be added without depending on this package. The
public entry points live in :mod:`giant_python.pipeline.extract`, which wraps
:mod:`giant_python.bandsilo.pipeline`.
"""
