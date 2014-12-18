
TODO
----

0. why (rapidswitch,graphic07) Bluebox/ (17728,35803) imgs
   -> cos 14711 multjoints U 2627 marine U 915 unsuits rm in 1 & not the other
      -> rsync graphic07 rapidswitch                                       PENDING
      -> rsync graphic07 protip                                            PENDING
   -> rm multJoints, marine from pipe-data in graphic and rapidswitch

1.1 remove multJoints, fusion marine, unsuits from pipe-data/Bluebox in all machines:
   * graphic06 not responding, still need do it
     ->       
   * 38482 total ims, 29887 total joints
   * 6116/29887 joints are multjoints ie 20%
   * 14711/38482 imgs are from a multjoint ie 38%
   -> can't rm all, let's be selective
      -> keep 1st images?
      	 -> joint-imgs.json dict should be useful
	 -> BlueboxMult, flick through the dirs to see when unsuit or zoomed
	    -> 2 imgs in joint:
	       unsuit is barcode, people's faces, super blur
	       -> (5,29)/50 of (img1,img2) were zoomed in (looked at imgs ~101979-106120)
	          (1,11)/50 of (img1,img2) were zoomed in (looked at imgs ~203469-98387)
	       -> (1,12)/50 of (img1,img2) were unsuit (looked at imgs ~101979-106120)
	          (1,10)/50 of (img1,img2) were unsuit (looked at imgs ~203469-98387)
            -> 4 imgs in joint:
	       ignored 1st 50, seemed to mostly be fusion marine
	       -> (8,6,6,8)/50 of (img1,img2,img3,img4) zoom in (looked at imgs 200093-98251)
	          (0,3,3,2)/50 of (img1,img2,img3,img4) unsuit (looked at imgs 200093-98251)
         * keep 1st img, ie remove (14711-6116)/38482 = 22%
	   -> those removed from pipe-data/Bluebox are removed_multjoin_imgs
	   -> dataset now 29887 imgs
   * 2679/38482 imgs are fusion marine ie 7%
     -> those removed from pipe-data/Bluebox are removed_fusionmarine_imgs
     -> dataset now 28743 (1535 already removed as multjoints)
   * 915/38482 imgs are unsuitable ie 2.4%
     -> those removed from pipe-data/Bluebox are removed_unsuit_imgs
     -> dataset now  ( already removed as multjoints or fusionmarine)

1.2 update train.txt and val.txt
   

3. train merged scrape/zone
   -> find proportion of ¬zone&scrape in test mismatched
   -> cp train.txt for scrape, flip ones flagged with zone to 1, train
   -> also, train zone only

4. train merged low/high soil risk
   -> find proportion of low risk in test mismatched
   -> cp train.txt for soil, flip ones flagged with low risk 1, train

*. more confident preds
   -> based on our 05-12-14 val set, HoT prices, assuming (!) 50p for 50%-75% conf
      -> water avgs  118p/img
      -> soil avgs   61p/img
      -> scrape avgs 51p/img
   -> relabel to confuse less
   -> remove from tran set to confuse less
   -> try multiple classes eg separate low/high risk
      -> low risk as target 0.7 => less money
   -> HoT: how to get overall conf for entire joint?

*. non sparse target vectors
   -> careful! could => less money



PAST: before 17-12-14
---------------------

manually go over scrape: stopped at ?? error ??

!! suggest saying no evidence when hatch markings not visible
very harsh, but could be v efficient. (except peelable or dirty)
suggest for human system to be able to say uncertain about scraping, then you'll
see how tough it is

retrain all using only single image joints, without Fusion marine
(multi-image likely neg so this is just undersampling)

scrape: should return flag if not all visible or no scrape on what's visible.
-> need correctly labelled no scrape zones
   -> 1) train a scrape zone netcom  

   -> 2) relabel misclassifications
-> how are accuracy, mislabelling when no hatch marks?
   -> 3) train a scrape net on no hatch marks only
         -> does it do better on those than if it also learned with the hatch 
	    marked cases? hopefully it can pick out that when fitting entry is
	    shinier, it's been scraped (which I think is true).
-> merge labels? separate nets?
   -> 4) train a merged label net (if no scrape zones then no scraping)
   -> 5) relabel misclassifications

train a hatch markings net:
-> if it gets high accuracy, replace scrape zone with it


soil:
-> include low risk in a smart way?


PAST: before CP demo
--------------------
-> try new inadcl_o solver mode
-> avoid Redbox Perfects
   -> run inadcl_o_noRedPerfs for 1000 on graphic05 to compare
   -> ok, should not avoid them
   
-> train inadcl_o more
   -> (79,63)%
   -> BIGGER BATCH SIZE
      -> didn't help
   -> initialise with clampdet net?
      -> and freeze backprop
   -> remove all NoClampUsed's?
   -> combine models

   
-> scrape_o
   -> iter1500 69.5% val
      -> just cant do better
   IMPROVE
   -> bigger batch size
      -> (72,78)%
   -> train/test only on {zs,z¬s,¬z¬s}
      -> looks shit, try exact same with normal data
   -> train/test only on {zs,z¬s} -> teaches scrape semantics
   -> combine models
02: train alexnet scrape fullTrain
    -> 
03: train alexnet scrape zones always visible
    -> 5000iter (63%,67%) 
05: train alexnet inadcl trainErr no CV!
06: train raznet scrape zones always visible
    -> 3000iter (61%,68%) trainErr no CV!
alexcaffe/task/scrape_o/none_fullTrain/train.log
alexcaffe/task/scrape_o/none/train.log   
   -> BLUEBOX ONLY! cos redbox blurs scrape marks!
      -> alexnet on graphic02, fullTrain
      -> raznet on graphic06, fullTrain

      
-> misal_o
   -> 200 lr4
      -> iter200 82% val

-> train soil_high_o
   -> 1000 lr4, 1000 lr5
      -> iter2000 83% val
   -> iter3300 (85,81)%

-> unsuit
   -> (89,92)%
   -> BIGGER BATCH
      cod ridiculous amplitude

-> train water_high_o
   -> 1000 lr4, 1000 lr5, 1000 lr6
      -> iter3000 (86,88)%  (84,80)%  (86,83)%
      -> iter2400 (82,87)%  (83,80)%
      -> iter1800 (84,88)%  (88,76)%    
      
-> graphic04 6 GB!!
   -> but compute capability (threads) is bottleneck

-> graphic02 batchsize 10!
   -> hence why soil good perf?
   
-> freeze backprop 
   -> inadcl_o: no, not overfitting
   -> misal_o: why not, smaller set
   
-> LOOK INTO TEST ITER
   -> run on graphic02
   -> val_18.txt vs val_19.txt
   -> with alexnet because light enough
   -> 
-> graphic02: 
-> eliminate barcode images with intersect wrongly classified imgs
   over all classes, and propose this as preprocessing step when we
   go to CP

Raz:
-> pool classifications for multi image queries
-> write preds.txt file - joint name not image name!!
demo flag names exact, say if merged


