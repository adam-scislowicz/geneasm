#!/usr/bin/env python

# Specifies the git_config parameter. This example does not provide Git credentials, so python SDK will try
# to use local credential storage.
git_config = {'repo': 'https://github.com/adam-scislowicz/geneasm.git',
              'branch': 'master'}
#             'commit': '4893e528afa4a790331e1b5286954f073b0f14a2'}

# In this example, the source directory 'pytorch' contains the entry point 'mnist.py' and other source code.
# and it is relative path inside the Git repo.
pytorch_estimator = PyTorch(entry_point='swaemb.py',
                            role='SageMakerRole',
                            source_dir='notebooks',
                            git_config=git_config,
                            instance_count=1,
                            instance_type='p3.8xlarge')