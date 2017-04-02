# Caffe_Helper

For those lazy guys who want to use Caffe in C++.

## Requirement
<ul>
<li>Caffe</li>
<li>Cuda</li>
</ul>

## Usage
```
$ helper prepare | train
$ helper test <u>test_file</u>
```

```
Caffe_Helper
├── Debug
│   └── src
├── nn
│   ├── model
│   ├── prototxt
│   └── training_set
│       ├── data
│       ├── data_mdb
│       ├── data_mdb(copy)
│       ├── label_mdb
│       ├── label_mdb(copy)
│       └── test_data
└── src
```

Binary file should be in Debug folder.
Unfortunately, the whole structure cannot be changed by parameters. This is a future work.
