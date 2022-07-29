#!/usr/bin/env python3

import grass.script as gscript

inputLidar= "{directory}/Kilmacahill_group1_densified_point_cloud_part_4.las"
partNo = "4"
res = 1
zrange = "-100,100"
# for now, running this program once to check z range, and then manually change this zrange, run he script again.
# someone should update this so that the zrange can be typed in by user after checking outliers without
# stopping the program or running the program twice

"""
https://grasswiki.osgeo.org/wiki/Processing_lidar_and_UAV_point_clouds_in_GRASS_GIS_(workshop_at_FOSS4G_Boston_2017)#Software
"""

def main():
    # start graphics monitor.
    gscript.run_command("d.mon",start="wx0")
    # print current computational region range
    gscript.run_command('g.region', flags='p')

    # steps to check reolution
    gscript.run_command("r.in.lidar",input=inputLidar,
                output="part%sbinnedcountRes%s"%(partNo,res),method="n", resolution=res,
                flags="en",overwrite=True)
    # print raster info in GRASS GIS window,
    # check # of points per cell to see if we have enough points at the set resolution to do surface interpolation later 
    gscript.run_command("r.info", map="part%sbinnedcountRes%s"%(partNo,res))
    # visualize histogram distribution of point count per cell
    gscript.run_command("d.histogram",map="part%sbinnedcountRes%s"%(partNo,res),flags="nc")

    # update computation region
    gscript.run_command('g.region', rast ="part%sbinnedcountRes%s"%(partNo,res),flags='p')
     
    # steps to check outliers by examining binned distribution of binned max and mean
    # (honestly, I don't understand why wee need to check both". I could have misunderstood the tutorial)
    # make DSM by binned max
    gscript.run_command("r.in.lidar",input=inputLidar,
                output="part%sBinnedMaxRes%s"%(partNo,res),method="max",resolution=res,
                flags="en",overwrite=True)
    gscript.run_command("r.report", map="part%sBinnedMaxRes%s"%(partNo,res),unit="c")
    # visualize binned max DSM, color-coded with elevation
    gscript.run_command("r.colors", map="part%sBinnedMaxRes%s"%(partNo,res),color="elevation",flags="e")
    gscript.run_command("d.mon",start="wx1")
    gscript.run_command("d.rast", map="part%sBinnedMaxRes%s"%(partNo,res))
    # make DSM by binned min
    gscript.run_command("r.in.lidar",input=inputLidar,
                output="part%sBinnedMinRes%s"%(partNo,res),method="min",resolution=res,
                flags="en",overwrite=True)
    gscript.run_command("r.report", map="part%sBinnedMinRes%s"%(partNo,res),unit="c")
    # visualize binned min DSM, color-coded with elevation
    gscript.run_command("r.colors", map="part%sBinnedMinRes%s"%(partNo,res),color="elevation",flags="e")
    gscript.run_command("d.mon",start="wx2")
    gscript.run_command("d.rast", map="part%sBinnedMinRes%s"%(partNo,res))


    # apply z filter and class filter and check the spatial distribution by looking at binned count
    # class filter = 2: only get ground class
    gscript.run_command("r.in.lidar",input=inputLidar,
                output="part%scount_ground%s"%(partNo,res),method="n",resolution=res,
                zrange = zrange, class_filter=2,
                flags="en", overwrite=True)
    gscript.run_command("r.report", map="part%scount_ground%s"%(partNo,res),unit="c")
    gscript.run_command("r.colors", map="part%scount_ground%s"%(partNo,res),color="viridis",flags="e")
    gscript.run_command("d.mon",start="wx3")
    gscript.run_command("d.rast", map="part%scount_ground%s"%(partNo,res))

    #------------------- Now start creating DTM-------------------
    
    # but first, set computational region
    gscript.run_command('g.region', rast ="part%sbinnedcountRes%s"%(partNo,res),flags='p')
    
    # import  .las format data as vector points in GeoPackage format.,
    # apply z-filtering to ignore outlying point clouds
    gscript.run_command("v.in.lidar",
                    input=inputLidar, output = "part%sPoints_griound"%partNo,class_filter=2,
                    flags="bt",zrange = zrange,overwrite=True)
    
    # interpolate the vectors using spline with tension
    gscript.run_command("v.surf.rst", input= "part%sPoints_griound"%partNo,tension = 25, smooth=1,
                    npmin=100, elevation="part%sDTM"%partNo, slope="part%sSlope"%partNo,
                    aspect="part%sAspect"%partNo, 
                    pcurvature="part%sprofile_curvature"%partNo,
                    tcurvature="part%stangential_curvature"%partNo,
                    mcurvatur="part%smean_curvature"%partNo, overwrite=True)
 
    # visualize the created DTM
    gscript.run_command("r.colors",map="part%sDTM"%(partNo),color="grey",flags="e")
    gscript.run_command("d.mon",start="wx4")
    gscript.run_command("d.rast", map="part%sDTM"%(partNo,))
    

    # create and visualie relief/hillshade given an illuminiation setting
    gscript.run_command("r.relief", input="part%sDTM"%partNo, output="part%sLidar135_20"%partNo, altitude=20,azimuth=135, overwrite=True)
    gscript.run_command("d.mon",start="wx5")
    gscript.run_command("d.rast", map="part%sLidar135_20"%partNo)


if __name__ == '__main__':
    main()
