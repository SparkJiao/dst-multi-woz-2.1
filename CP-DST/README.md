# Enhanced Multi-Domain Dialogue State Tracker with Second-Order Slot Interaction

This is the original pytorch implementation of **CP-DST**. Some codes are borrowed from **SUMBT**.

### Data prepration & pre-procesisng
* Download corpus
  * MultiWOZ 2.1: [download](http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/)
    * Note: our experiments conducted on MultiWOZ 2.1 corpus
* Pre-process corpus
  * The download original corpus are loacated in ``data/multiwoz2.1_5/original``
  * See ``data/multiwoz2.1_5/original/convert_to_glue_format_5domains_full_value.py``
  * The pre-processed data are located in ``data/multiwoz2.1_5/``

### Train CP-DST and Experiments
Please see ``scripts/cls_graph2_plus/run-data2.1-cls-graph2p-1.4-7.14.sh``

For zero-shot experiments please see ``scripts/cls_graph2_plus/run-data2.1-cls-graph2p-1.4-7.14-$domain.sh``

``$domain`` can be ``attraction/hotel/restaurant/taxi/train``.

For value supervision experiments, please see ``scripts/cls_graph2_plus/run-data2.1-cls-graph2p-1.4-7.14-vsup=$x.sh``.

For masked slot attention experiments, please see ``scripts/cls_graph2_plus/run-data2.1-cls-graph2p-1.4-7.14-mask-$domain.sh``.