"""Explicit histogram regions, fixed independently of network assignments.
These regions are only for descriptive statistics, not iterative reassignments.
"""
import numpy as np

def density_regions(raw):
    # A large online-share gap separates digital customers from store shoppers.
    digital=raw.online_share.to_numpy()>.6
    # Among store shoppers, large baskets are a distinct, infrequent-visit mode.
    premium=(~digital)&(raw.avg_basket.to_numpy()>90.)
    # The two remaining modes have annual-spend peaks around $1,400 and $4,000.
    frequent=(~digital)&(~premium)&(raw.annual_spend.to_numpy()>=2500.)
    budget=(~digital)&(~premium)&(~frequent)
    return np.stack([budget,frequent,digital,premium],axis=1)

def region_statistics(raw,x):
    regions=density_regions(raw)
    assert np.all(regions.sum(axis=1)==1)
    means=np.stack([x[m].mean(axis=0,dtype=np.float64) for m in regions.T])
    variances=np.stack([x[m].var(axis=0,dtype=np.float64) for m in regions.T])
    return regions,means,variances
