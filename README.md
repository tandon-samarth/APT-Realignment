# APT-Realignment

## Objective 
Currently, the highest geocoding accuracy we can offer as a map provider is an Address Point (APT) i.e. an x,y location on a map that represents an address. However, we have observed that the location of an APT is not always accurate and not necessarily always placed on the the Building Footprint (BPF) for APTs of type "Building". Hence we are targetting to improve Address points Positional accuracy for APT's to lie on building Foot print.

## Approach 
We are targeting USA and within USA we are targeting 5 counties.The approach is based on using Land Parcels from DMP Source and Building footprint for targetted state from Microsoft. We use MDS data to extract APT information and use sjoin to process data.

![Image text](https://github.com/tandon-samarth/APT-Realignment/blob/main/realignment_pipeline.JPG) 
