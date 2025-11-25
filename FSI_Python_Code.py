import arcpy
from arcpy.ia import Foreach
import numpy as np
from scipy.stats import linregress
from itertools import chain
import os
import logging
import traceback

#
#Deletes fields for Feature Class except the fields given as a parameter
#
def deleteUnnceesaryFields(inFeature, xfields):
    toBeDeletedFields = []
    allFields = arcpy.ListFields(inFeature)
    
    for f in allFields:
        if not (f.name in xfields):
            toBeDeletedFields.append(f.name)
            
    if(len(toBeDeletedFields)> 0):
        arcpy.management.DeleteField(inFeature, toBeDeletedFields)


#
#Inverts given raster and saves inverted one with '_inv' suffix
#
def invert_raster(rasterX):
    resultDesc = arcpy.management.GetRasterProperties(rasterX,"MAXIMUM")
    rasterMaxValue = resultDesc.getOutput(0)
    resultDesc = arcpy.management.GetRasterProperties(rasterX,"MINIMUM")
    rasterMinValue = resultDesc.getOutput(0)

    #print("rasterMaxValue:"+rasterMaxValue+" rasterMinValue:"+rasterMinValue)

    rasterX = arcpy.sa.Raster(rasterX)
    rasterInverted = ((rasterX - float(rasterMaxValue)) * -1) + float(rasterMinValue)
    rasterInverted.save(working_full_directory+"/"+str(rasterX)+"_inv")


##############
# Settings
#
##############
#Main directory that contains GDB and results  
working_directory = r"D:\Projects\FSI_PYTHON\FSI_Python"
#Output directory that contains raster results
working_directory_output = working_directory + "\\Output"
#File Geodatabase that contains all raster input layers
fgdb_name = "FSI_inputs.gdb"
#Feature layer that contains points
feature_point = "Flood_and_NonFlood_training"
#Feature layer that contains points for FSI Normalization
feature_point_FSI_norm = "Flood_and_NonFlood_validation"
#default fields in feature point
def_fields = ['OBJECTID','Shape','Sample_ID','Flood_Samp','FS1NOFS0']
#Raster input layer those organized as group. Layer names SHOULD BE unique and
#and DO NOT EXCEED 30 chars in the name
raster_groups = [
        ["API_mm_daily_ENACTS1991_2020","Precipitation_mm_annual_ENACTS1983_2017","Precipitation_mm_daily_ENACTS1991_2020","Surface_temperature_oC_mean_annual_ENACTS1983_2017","Surface_temperature_oC_mean_daily_ENACTS1991_2020"],
        ["Altitude","Altitude_max_within_4x4_matrix","Slope_perc","Slope_perc_max_within_4x4_matrix","TRI","TWI","Aspect","Curvature","Drainage_density","Flow_acc","LS","Plan_curvature","Profile_curvature","SPI","TPI"],
        ["Lithology","Soiltexture","K_factor_Soil_permeability_erodibirity","Soil_depth"],
        ["Landcover","NDVI","NDWI","TNDVI"],
        ["Distance_from_Faults_and_earthquake_hotspots1950_2022"],
        ["Distance_from_road"],
        ["Distance_from_river"]
    ]
#Maximum p value for a layer that will not excluded in group
maxPValue = 0.001

##############
# Initialize
#
##############
if(os.path.isdir(working_directory_output) == False):
    os.makedirs(working_directory_output)
       
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(working_directory_output + '/debug.log',mode='w'),
        logging.StreamHandler()
    ]
)

try:
    #Checking out Spatial Analyst license for Concurrent License environmet
    if arcpy.CheckExtension("Spatial") == "Available":
        logging.debug("Spatial license is Available")  
        arcpy.CheckOutExtension("Spatial")
        logging.debug("Checked Out Extension!") 
    else:
        # raise a custom exception
        logging.error("ERROR: Spatial license is NOT Available!!!")
        raise LicenseError

    working_full_directory = working_directory + "\\" + fgdb_name
    arcpy.env.workspace = working_full_directory
    arcpy.env.overwriteOutput = True

    raster_layers = list(chain.from_iterable(raster_groups))



    raster_field_names = []
    for r in raster_layers:
        raster_field_names.append(r+"_1")

    all_raster_extract_field_names = []
    for r in raster_layers:
        all_raster_extract_field_names.append(r+" "+r+"_1")

    #Delete old columns in feature layer
    deleteUnnceesaryFields(feature_point,def_fields)
    
    arcpy.sa.ExtractMultiValuesToPoints(in_point_features=feature_point, 
                                        in_rasters=";".join(all_raster_extract_field_names),
                                        bilinear_interpolate_values="NONE")

    ##############
    # Correlation
    # r and P values
    ##############
    logging.debug("Start Correlation")

    array_y = []
    with arcpy.da.SearchCursor(feature_point,["FS1NOFS0"]) as cur:
        for row in cur:
            array_y.append(row[0])

    ray = arcpy.da.TableToNumPyArray(feature_point, raster_field_names)

    resultCorrelationNp = np.zeros(len(raster_layers), dtype=[('Features', 'U256'), ('corr', 'float'), ('sign', 'float')])
    
    isInvertedLayer = False
    for index, item in enumerate(raster_layers):
        x = np.array(ray[raster_field_names[index]])
        y = np.array(array_y)
        res = linregress(y, x)
        resultCorrelationNp[index]['Features'] = raster_field_names[index]
        resultCorrelationNp[index]['corr'] = res.rvalue
        resultCorrelationNp[index]['sign'] = res.pvalue

    # DELETE
    # if(item == "Altitude_Clip_2"):
        # resultCorrelationNp[index]['corr'] = -0.1234
        # resultCorrelationNp[index]['sign'] = 0.00123
    
    #Checking for inverting
        if(resultCorrelationNp[index]['corr'] < 0):
            isInvertedLayer = True
            logging.debug("inverting raster "+str(item))
            invert_raster(item)
            layerNewName = str(item) + "_inv"
            raster_layers[index] = layerNewName
            
            for indexG, itemG in enumerate(raster_groups): 
                for ix, itemX in enumerate(itemG):
                    if(itemX == str(item)):
                        raster_groups[indexG][ix] = layerNewName
                        break

    #If there is an inverted layer, we have to do calculation from begining!
    if (isInvertedLayer):
        logging.debug("There is an inverted raster. Recalculating linregress")
        
        raster_field_names = []
        for r in raster_layers:
            raster_field_names.append(r+"_1")
            
        #delete old columns in input point feature
        deleteUnnceesaryFields(feature_point,def_fields)

        all_raster_extract_field_names = []
        for r in raster_layers:#this raster_layer variable holds latest layer name even inverted
            all_raster_extract_field_names.append(r+" "+r+"_1")
            
        arcpy.sa.ExtractMultiValuesToPoints(in_point_features=feature_point,
                                            in_rasters=";".join(all_raster_extract_field_names),
                                            bilinear_interpolate_values="NONE")

        ray = arcpy.da.TableToNumPyArray(feature_point, raster_field_names)

        resultCorrelationNp = np.zeros(len(raster_layers), dtype=[('Features', 'U256'), ('corr', 'float'), ('sign', 'float')])
    
        for index, item in enumerate(raster_layers):
            x = np.array(ray[raster_field_names[index]])
            y = np.array(array_y)
            res = linregress(y, x)
            resultCorrelationNp[index]['Features'] = raster_field_names[index]
            resultCorrelationNp[index]['corr'] = res.rvalue
            resultCorrelationNp[index]['sign'] = res.pvalue

    np.savetxt(working_directory+"/result_correlation.csv", resultCorrelationNp, fmt='%s, %.70f, %.70f' ,delimiter=",",header="Features,Pearson correlation (r),P value (significance)")
    logging.debug("End Correlation")
    # print("===============")
    # for index, item in enumerate(raster_layers):
    #     print("item:",item)
    # print("===============")
    # for indexG, itemG in enumerate(raster_groups): 
    #     print("~~~~~~~")
    #     for ix, itemX in enumerate(itemG):
    #         print("itemX:",itemX)


    ##############
    # Linear regression
    # eg: c3, c2 calculations
    ##############
    logging.debug("Start Linear regression")
    result = []
    for index, item in enumerate(raster_field_names):
        #print("index:",index, "item:",item)
        x = np.array(ray[item])
        y = np.array(array_y)
        z = np.polyfit(x,y, 3)
        result.append([str(item),z[0],z[1],z[2],z[3]])

    npResult = np.array(result)
 
    resultLinearNp = np.zeros(len(result), dtype=[('Features', 'U256'), ('c3', 'float'), ('c2', 'float'), ('c1', 'float'), ('b', 'float')])
    resultLinearNp['Features'] = raster_layers#npResult[:, 0]
    resultLinearNp['c3'] = npResult[:, 1]
    resultLinearNp['c2'] = npResult[:, 2]
    resultLinearNp['c1'] = npResult[:, 3]
    resultLinearNp['b'] = npResult[:, 4]
 
    np.savetxt(working_directory+"/result_linear.csv", resultLinearNp, fmt='%s, %.11f, %.11f, %.11f, %.11f' ,delimiter=",",header="Features,c3,c2,c1,b")

    for index, item in enumerate(raster_layers):
        val = result[index]
        c3 = val[1]
        c2 = val[2]
        c1 = val[3]
        b = val[4]
        logging.debug(" ".join(["processing item",item,"c3",str(c3),"c2",str(c2),"c1",str(c1),"b",str(b)]))
        myRaster = arcpy.sa.Raster(item)

        # FSIf Calculation
        rasterOut = 1 / (1 + arcpy.sa.Exp((c3 * (myRaster ** 3) + (c2 * myRaster ** 2) + (c1 * myRaster + b)) * -1))

        rasterOut.save(working_directory_output +"\\"+ item+"_Factor.tif")
        logging.debug("Saved raster "+item)

    logging.debug("End Linear regression")

    ##############
    # Finding FSI by combining all FSIf into one raster layer
    # Formula number 4, document page 3
    ##############
    logging.debug("Start Finding FSI")
    inRaster = arcpy.Raster(raster_groups[0][0])
    rasterTotal = arcpy.Raster(inRaster.getRasterInfo())

    cnt = 0
    myloopR = 0
    for rg in raster_groups:
        logging.debug("===New Group====")
        inRaster = arcpy.Raster(raster_groups[0][0])
        rasterGroup = arcpy.Raster(inRaster.getRasterInfo())
        myloop = 0
       
        numOfGroupElement = 0
        for r in rg:
            inraster = arcpy.sa.Raster(working_directory_output +"\\"+ r +"_Factor.tif")
            valueR = 0.0
            excludeLayer = False
            for item2 in resultCorrelationNp:
                if(item2[0] == r + "_1" or item2[0] == r):
                    if(float(item2[2]) >= maxPValue):
                        logging.debug("Layer "+r+"_Factor.tif has greater then MAX_P value, removed from group! It's value:"+str(item2[2]))
                        excludeLayer = True
                        break
                    valueR = item2[1]
                    numOfGroupElement += 1
                    break
            
            if(excludeLayer):
                continue
            if(myloop==0):
                rasterGroup = (inraster *  valueR)
            else:
                rasterGroup += (inraster *  valueR)
            myloop+=1
            logging.debug("\\"+ r +"_Factor.tif * " + str(valueR)+" <- R-value")
        
        #At least one layer element should be processed for creating result layers
        if(numOfGroupElement > 0):
            rasterGroup = arcpy.sa.Divide(rasterGroup,numOfGroupElement)
            rasterGroup.save(working_directory_output +"\\rr_grp"+str(cnt)+".tif") 
            logging.debug("Group Output:" + working_directory_output +"\\rr_grp"+str(cnt)+".tif LEN:"+str(numOfGroupElement))
            if(myloopR==0):
                rasterTotal = rasterGroup
            else:
                rasterTotal += rasterGroup
            myloopR +=1
        else:
            logging.debug("No layer in this Group")   
        cnt+=1
    logging.debug("End Finding FSI")

    
    logging.debug("Result raster is dividing to "+str(myloopR)+" and saving...")
    rasterTotal.save(working_directory_output +"\\FSI_INDEX.tif")
    
    if (myloopR == 0):
        logging.info("No group result is produced, Total result is empty!")
        exit()

    logging.info("Num. of group result is "+str(myloopR)+", check Total Result")
    
    ##############
    # FSI Normalization
    # 
    ##############
    logging.debug(" ")
    logging.debug("Start FSI Normalization")
    
    fsi_max = float(arcpy.management.GetRasterProperties(rasterTotal,"MAXIMUM").getOutput(0))
    fsi_min = float(arcpy.management.GetRasterProperties(rasterTotal,"MINIMUM").getOutput(0))

    logging.debug("MINIMUM value " +str(fsi_min))
    logging.debug("MAXIMUM value " +str(fsi_max))
    logging.debug("MEAN value " +arcpy.management.GetRasterProperties(rasterTotal,"MEAN").getOutput(0))
    logging.debug("STD value " + arcpy.management.GetRasterProperties(rasterTotal,"STD").getOutput(0))
    
    #normalizedFSI = arcpy.Raster(rasterTotal.getRasterInfo())
    xa = rasterTotal - fsi_min
    
    FSI_Normalized = arcpy.sa.Divide(xa, (fsi_max - fsi_min))
    FSI_Normalized.save(working_directory_output +"\\FSI_Normalized.tif")
    logging.debug("FSI Normalization finished, check FSI_Normalized.tif")

    ##############
    # Correlation with FSI Normalization
    # r and P values
    ##############
    logging.debug(" ")
    logging.debug("Start Linear regression with FSI Normalization")
    
    #Delete old columns in feature layer
    deleteUnnceesaryFields(feature_point_FSI_norm,def_fields)
    
    arcpy.sa.ExtractMultiValuesToPoints(in_point_features=feature_point_FSI_norm, 
                                    in_rasters=[[working_directory_output +"\\FSI_Normalized.tif" , 'FSI_Normalized_1']],
                                    bilinear_interpolate_values="NONE")

    raster_field_names = ['FSI_Normalized_1']
    ray = arcpy.da.TableToNumPyArray(feature_point_FSI_norm, raster_field_names)

    raster_layers = ['FSI_Normalized']
    
    array_y = []
    with arcpy.da.SearchCursor(feature_point,["FS1NOFS0"]) as cur:
        for row in cur:
            array_y.append(row[0])
            

    result = []
    for index, item in enumerate(raster_field_names):
        #print("index:",index, "item:",item)
        x = np.array(ray[item])
        y = np.array(array_y)
        z = np.polyfit(x,y, 3)
        result.append([str(item),z[0],z[1],z[2],z[3]])

    npResult = np.array(result)
 
    resultLinearNp = np.zeros(len(result), dtype=[('Features', 'U256'), ('c3', 'float'), ('c2', 'float'), ('c1', 'float'), ('b', 'float')])
    resultLinearNp['Features'] = raster_layers#npResult[:, 0]
    resultLinearNp['c3'] = npResult[:, 1]
    resultLinearNp['c2'] = npResult[:, 2]
    resultLinearNp['c1'] = npResult[:, 3]
    resultLinearNp['b'] = npResult[:, 4]
 
    np.savetxt(working_directory+"/result_linear_normalizedFSI.csv", resultLinearNp, fmt='%s, %.11f, %.11f, %.11f, %.11f' ,delimiter=",",header="Features,c3,c2,c1,b")

    for index, item in enumerate(raster_layers):
        val = result[index]
        c3 = val[1]
        c2 = val[2]
        c1 = val[3]
        b = val[4]
        logging.debug(" ".join(["processing item",item,"c3",str(c3),"c2",str(c2),"c1",str(c1),"b",str(b)]))
        myRaster = arcpy.sa.Raster(working_directory_output +"\\"+item+".tif") #previous calculations are doing at GDB but this is pointing a file!

        # FSIf Calculation
        rasterOut = 1 / (1 + arcpy.sa.Exp((c3 * (myRaster ** 3) + (c2 * myRaster ** 2) + (c1 * myRaster + b)) * -1))

        rasterOut.save(working_directory_output +"\\"+ item+"_Factor.tif")
        logging.debug("Saved raster "+item+ " as "+item+"_Factor.tif")

    logging.debug("End Linear regression")
    
except Exception as ex:
    logging.error(traceback.format_exc())
    logging.error(ex)

#Release Extension
arcpy.CheckInExtension("Spatial")
