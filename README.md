# CBPlus
 
Structure of results directory.
```bash
├results/
├── merged_network_results
│   ├── ANY
│   ├── ATW
│   │   └── tw_1d
│   │       ├── co-retweet
│   │       │   └── overlapping
│   │       │       ├── edge_list
│   │       │       │   └── temporal
│   │       │       ├── info_edge_list
│   │       │       │   └── temporal
│   │       │       └── th_0.7
│   │       │           ├── analysis
│   │       │           ├── community
│   │       │           ├── edge_list
│   │       │           └── graph
│   │       ├── info_tw
│   │       │   ├── end_date_list.p
│   │       │   └── start_date_list.p
│   │       └── multi_co_action
│   │           └── co-retweet_overlapping_th_0.7_co-reply_overlapping_th_0.7co-url-domain_overlapping_th_0.7
│   │               ├── analysis
│   │               └── community
│   │                   ├── coms
│   │                   ├── gephi_graph
│   │                   ├── graph
│   │                   └── user_dataframe
│   └── OTW
└── temporal_network_results
    ├── ANY
    ├── ATW
    └── OTW