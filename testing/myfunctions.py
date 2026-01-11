import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
import json

from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear, sie, sis

tfd = tfp.distributions

def makeJson(phys_model):
    lensMasses, lensLights, sourceLights = phys_model.getProfiles()
    masspriors = []
    lenspriors = []
    sourcepriors = []
    
    for ii in lensMasses:
        paramDict = dict.fromkeys(ii._params)
        for iii in paramDict.keys():
            if iii == "gamma":
                paramDict[iii] = dict(mean="CHANGEME", stdev="CHANGEME",lowerBound= "CHANGEME",upperBound= "CHANGEME")
            else:
                paramDict[iii] = dict(mean="CHANGEME", stdev="CHANGEME")
                
        objectDict = {"name": ii._name, "params": paramDict}
        masspriors.append(objectDict)
    
    for ii in lensLights:
        paramDict = dict.fromkeys([x for x in ii._params if x != "amp"])
        for iii in paramDict.keys():
            if "n_sersic" in iii:
                paramDict[iii] = dict(lowerBound= "CHANGEME",upperBound= "CHANGEME")
            elif "e1" in iii or "e2" in iii:
                paramDict[iii] = dict(mean="CHANGEME", stdev="CHANGEME",lowerBound= "CHANGEME",upperBound= "CHANGEME")
            else:
                paramDict[iii] = dict(mean="CHANGEME", stdev="CHANGEME")
                
        objectDict = {"name": ii._name, "params": paramDict}
        lenspriors.append(objectDict)
        
    for ii in sourceLights:
        paramDict = dict.fromkeys([x for x in ii._params if x != "amp"])
        for iii in paramDict.keys():
            if "n_sersic" in iii:
                paramDict[iii] = dict(lowerBound= "CHANGEME",upperBound= "CHANGEME")
            elif "e1" in iii or "e2" in iii:
                paramDict[iii] = dict(mean="CHANGEME", stdev="CHANGEME",lowerBound= "CHANGEME",upperBound= "CHANGEME")
            else:
                paramDict[iii] = dict(mean="CHANGEME", stdev="CHANGEME")
                
        objectDict = {"name": ii._name, "params": paramDict}
        sourcepriors.append(objectDict)
    
    
    prior = [masspriors, lenspriors, sourcepriors]
    with open("prior.json", "w") as outfile:
        json.dump(prior, outfile, indent=4)
            

def readJson(path, shapelet_n_max=4):
    file = open(path)
    priordict = json.load(file)
    
    lens_prior = []
    lens_light_prior = []
    source_light_prior = []
    
    lens_mass_model = []
    lens_light_model_Forward = []
    source_light_model_Forward = []
    
    lens_light_model_Backward = []
    source_light_model_Backward = []
    
    for ii in priordict[0]:
        
        if ii["name"] == "EPL":
            lens_mass_model.append(epl.EPL(50))
        elif ii["name"] == "SHEAR":
            lens_mass_model.append(shear.Shear())
        elif ii["name"] == "SIE":
            lens_mass_model.append(sie.SIE())
        elif ii["name"] == "SIS":
            lens_mass_model.append(sis.SIS())
            
        paramDict = {}
        for iii in ii["params"].keys():
            currentParameter = ii["params"][iii]
            if "theta_E" in iii:
                try:
                    paramDict[iii] = tfd.TruncatedNormal(currentParameter["mean"],currentParameter["stdev"],currentParameter["lowerBound"],currentParameter["upperBound"])
                except:
                    paramDict[iii] = tfd.LogNormal(jnp.log(currentParameter["mean"]),currentParameter["stdev"])
            elif "gamma" in iii and ii["name"] != "SHEAR":
                paramDict[iii] = tfd.TruncatedNormal(currentParameter["mean"],currentParameter["stdev"],currentParameter["lowerBound"],currentParameter["upperBound"])
            elif "e1" in iii or "e2" in iii:
                try:
                    paramDict[iii] = tfd.TruncatedNormal(currentParameter["mean"],currentParameter["stdev"],currentParameter["lowerBound"],currentParameter["upperBound"])
                except:
                    pass
            else:
                paramDict[iii] = tfd.Normal(currentParameter["mean"],currentParameter["stdev"])
                
        lens_prior.append(tfd.JointDistributionNamed(paramDict))
    lens_prior = tfd.JointDistributionSequential(lens_prior)
    
    for ii in priordict[1]:
        if ii["name"] == "SERSIC":
            lens_light_model_Backward.append(sersic.Sersic(use_lstsq=True))
        elif ii["name"] == "SERSIC_ELLIPSE":
            lens_light_model_Backward.append(sersic.SersicEllipse(use_lstsq=True))
        elif ii["name"] == "CORE_SERSIC":
            lens_light_model_Backward.append(sersic.CoreSersic(use_lstsq=True))
        elif ii["name"] == "SHAPELETS":
            lens_light_model_Backward.append(shapelets.Shapelets(ii["n_max"], interpolate=False, use_lstsq=True))
            
        if ii["name"] == "SERSIC":
            lens_light_model_Forward.append(sersic.Sersic(use_lstsq=False))
        elif ii["name"] == "SERSIC_ELLIPSE":
            lens_light_model_Forward.append(sersic.SersicEllipse(use_lstsq=False))
        elif ii["name"] == "CORE_SERSIC":
            lens_light_model_Forward.append(sersic.CoreSersic(use_lstsq=False))
        elif ii["name"] == "SHAPELETS":
            lens_light_model_Forward.append(shapelets.Shapelets(ii["n_max"], interpolate=False, use_lstsq=False))
            
        paramDict = {}
        for iii in ii["params"].keys():
            currentParameter = ii["params"][iii]
            if "R_sersic" in iii:
                paramDict[iii] = tfd.LogNormal(jnp.log(currentParameter["mean"]),currentParameter["stdev"])
            elif "n_sersic" in iii:
                paramDict[iii] = tfd.Uniform(currentParameter["lowerBound"],currentParameter["upperBound"])
            elif "e1" in iii or "e2" in iii:
                paramDict[iii] = tfd.TruncatedNormal(currentParameter["mean"],currentParameter["stdev"],currentParameter["lowerBound"],currentParameter["upperBound"])
            else:
                paramDict[iii] = tfd.Normal(currentParameter["mean"],currentParameter["stdev"])
            if "beta" in iii:
                paramDict[iii] = tfd.LogNormal(jnp.log(currentParameter["mean"]),currentParameter["stdev"])
                
        lens_light_prior.append(tfd.JointDistributionNamed(paramDict))
    lens_light_prior = tfd.JointDistributionSequential(lens_light_prior)
    
    for ii in priordict[2]:
        
        if ii["name"] == "SERSIC":
            source_light_model_Backward.append(sersic.Sersic(use_lstsq=True))
        elif ii["name"] == "SERSIC_ELLIPSE":
            source_light_model_Backward.append(sersic.SersicEllipse(use_lstsq=True))
        elif ii["name"] == "CORE_SERSIC":
            source_light_model_Backward.append(sersic.CoreSersic(use_lstsq=True))
        elif ii["name"] == "SHAPELETS":
            source_light_model_Backward.append(shapelets.Shapelets(ii["n_max"], interpolate=False, use_lstsq=True))
        
        if ii["name"] == "SERSIC":
            source_light_model_Forward.append(sersic.Sersic(use_lstsq=False))
        elif ii["name"] == "SERSIC_ELLIPSE":
            source_light_model_Forward.append(sersic.SersicEllipse(use_lstsq=False))
        elif ii["name"] == "CORE_SERSIC":
            source_light_model_Forward.append(sersic.CoreSersic(use_lstsq=False))
        elif ii["name"] == "SHAPELETS":
            source_light_model_Forward.append(shapelets.Shapelets(ii["n_max"], interpolate=False, use_lstsq=False))
        
        paramDict = {}
        for iii in ii["params"].keys():
            currentParameter = ii["params"][iii]
            if "R_sersic" in iii:
                paramDict[iii] = tfd.LogNormal(jnp.log(currentParameter["mean"]),currentParameter["stdev"])
            elif "n_sersic" in iii:
                paramDict[iii] = tfd.Uniform(currentParameter["lowerBound"],currentParameter["upperBound"])
            elif "e1" in iii or "e2" in iii:
                paramDict[iii] = tfd.TruncatedNormal(currentParameter["mean"],currentParameter["stdev"],currentParameter["lowerBound"],currentParameter["upperBound"])
            else:
                paramDict[iii] = tfd.Normal(currentParameter["mean"],currentParameter["stdev"])
            if "beta" in iii:
                paramDict[iii] = tfd.LogNormal(jnp.log(currentParameter["mean"]),currentParameter["stdev"])
        
        source_light_prior.append(tfd.JointDistributionNamed(paramDict))
    source_light_prior = tfd.JointDistributionSequential(source_light_prior)
    
    prior = tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])
    
    phys_model_Backward = PhysicalModel(lens_mass_model, lens_light_model_Backward,source_light_model_Backward)
    phys_model_Forward = PhysicalModel(lens_mass_model, lens_light_model_Forward,source_light_model_Forward)
    
    SingleSources = [];
    for ii in source_light_model_Forward:
        SingleSources.append(PhysicalModel(lens_mass_model, lens_light_model_Forward, [ii]))
    
    return prior, phys_model_Backward, phys_model_Forward, SingleSources
        
    
def getPriors(phys_model):
    lensMasses, lensLights, sourceLights = phys_model.getProfiles()
    
    lmass = []
    llights = []
    slights = []
    for ii in lensMasses:
        if ii._name == "EPL":
            lmass.append("EPL")
        elif ii._name == "SIE":
            lmass.append("SIE")
        elif ii._name == "SIS":
            lmass.append("SIS")
        elif ii._name == "SHEAR":
            lmass.append("SHEAR")
    for ii in lensLights:
        if ii._name == "SERSIC":
            llights.append("SERSIC")
        elif ii._name == "SERSIC_ELLIPSE":
            llights.append("SERSIC_ELLIPSE")
        elif ii._name == "CORE_SERSIC":
            llights.append("CORE_SERSIC")
        elif ii._name == "SHAPELETS":
            llights.append("SHAPELETS")
    for ii in sourceLights:
        if ii._name == "SERSIC":
            slights.append("SERSIC")
        elif ii._name == "SERSIC_ELLIPSE":
            slights.append("SERSIC_ELLIPSE")
        elif ii._name == "CORE_SERSIC":
            slights.append("CORE_SERSIC")
        elif ii._name == "SHAPELETS":
            slights.append("SHAPELETS")
    return lmass, llights, slights


def countParameters(phys_model):
    lensMasses, lensLights, sourceLights = phys_model.getProfiles()
    
    count = 0
    for ii in lensMasses:
        count = count + len([x for x in ii._params if "amp" not in x])
    for ii in lensLights:
        count = count + len([x for x in ii._params if "amp" not in x])
    for ii in sourceLights:
        count = count + len([x for x in ii._params if "amp" not in x])    
    return count

def printToFile(thing, path):
    outputfile = open(path, 'a')
    print(thing)
    print(thing,file=outputfile)
    outputfile.close()
    
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def saveParametersAsJson(params,path,digits):
    for ii in range(0,len(params)):
        for iii in range(0,len(params[ii])):
            for iv in params[ii][iii].keys():
                params[ii][iii][iv] = np.trunc(np.round(float(params[ii][iii][iv]),3),3)
    with open(path, "w") as outfile:
        json.dump(params, outfile, indent=4)

