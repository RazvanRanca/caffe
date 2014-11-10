
TODO:
-> scrape, improve
-> inadcl, improve
-> undersample bluebox not to 90 but to min(90,red_imb)!
   othw can't bring all redboxes in


max batch: train_batch, val_batch (test_iter)
06: 10,5
05: 8,8
04:            | 1,10 (80)
03: 5,1
02: 11,2 (322)
01:

Alex:
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


