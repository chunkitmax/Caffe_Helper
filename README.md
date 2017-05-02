# caffeine

For those lazy guys who want to use Caffe in C++.

Multiple labels support. (Test needed)

## Requirement
<ul>
<li>Caffe</li>
<li>Cuda</li>
</ul>

## Usage
```
$ caffeine [data/image] prepare | train
$ caffeine [data/image] test {test_file_name}
$ caffeine help
```

```
.
└── nn
    ├── model
    │   ├── ?.caffemodel
    │   └── ?.solverstate
    ├── prototxt
    │   ├── mean.binaryproto
    │   ├── net.prototxt
    │   ├── solver.prototxt
    │   └── test.prototxt
    └── training_set
        ├── data
        │   └── ?.jpg ...
        ├── dataList.txt
        ├── data_mdb
        │   ├── data.mdb
        │   └── lock.mdb
        ├── label_mdb
        │   ├── data.mdb
        │   └── lock.mdb
        ├── test_data
        │   └── ?.jpg ...
        ├── dataList.txt
        ├── test_data_mdb
        │   ├── data.mdb
        │   └── lock.mdb
        └── test_label_mdb
            ├── data.mdb
            └── lock.mdb
```

Unfortunately, the whole structure cannot be changed by parameters. This is a future work.
