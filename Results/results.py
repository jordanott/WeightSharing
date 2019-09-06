#!/usr/bin/env python
# coding: utf-8

# # Results

# In[1]:


import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from helper import *
from tqdm import tqdm

pbar = tqdm(total=19)
# In[2]:

from IPython.core.pylabtools import figsize
figsize(20, 6)


# In[3]:


# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=20)
# matplotlib.rcParams['lines.linewidth'] = lw


# # MNIST
# ---

# In[6]:


mnist_all = pd.read_csv('mnist.csv')
mnist = mnist_all[mnist_all['aug_type'] != 'vcp']

mnist_cnn = mnist[mnist['model'] == 'cnn'].reset_index()
mnist_fcn = mnist[mnist['model'] == 'fcn'].reset_index()
mnist_fcn_vcp = mnist_all[mnist_all['aug_type'] == 'vcp'].reset_index()


# In[7]:


convert_table(mnist, eval_type='translation', to_latex=False)


# In[8]:


convert_table_vcp(mnist_fcn_vcp)


# # CIFAR
# ---

# In[9]:


cifar_all = pd.read_csv('cifar.csv').append(pd.read_csv('cifar_1.csv'), ignore_index=True)
cifar = cifar_all[cifar_all['aug_type'] != 'vcp']

cifar_cnn = cifar[cifar['model'] == 'cnn'].reset_index()
cifar_fcn = cifar[cifar['model'] == 'fcn'].reset_index()
cifar_fcn_vcp = cifar_all[cifar_all['aug_type'] == 'vcp'].reset_index()


# In[10]:


convert_table(cifar, eval_type='val_acc', to_latex=False)


# In[11]:


convert_table_vcp(cifar_fcn_vcp)


# # Paper Figures
# ---
# ### 1. Translation (left: FCN; right: CNN)
#     * Translation Augmented Training Acc
#     * Un-Augmented Validation Acc
#     * Translation Augmented Validation Acc

# In[13]:


# translation_paper_fig(mnist_fcn, mnist_cnn)
# pbar.update(1)

# In[14]:


translation_paper_fig(cifar_fcn, cifar_cnn, cifar=True)
pbar.update(1)

# ### 2. Approximate Weight Sharing (left: distance @ 1; right: distance @ 4)

# In[15]:


# approx_ws(mnist_fcn, dist_type='dist', y=[.05, .12])
#approx_ws(mnist_fcn, dist_type='cos')
# pbar.update(1)

# In[16]:


#approx_ws(cifar_fcn, dist_type='dist')
approx_ws(cifar_fcn, dist_type='cos',y=[.85, 1],cifar=True)
pbar.update(1)

# ### 3. Swap (left: MNIST; right: CIFAR)

# In[17]:

plt.clf(); _=plt.figure(figsize=(10, 6));
# plt.subplot(1,2,1)
p = sns.color_palette('coolwarm', 11)
sns.lineplot(x='epoch', y='swap', data=mnist_cnn, label='CNN',color=p[0]); sns.lineplot(x='epoch', y='swap', data=mnist_fcn, label='FCN',color=p[-1])
plt.title('Feature Swap MNIST'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(shadow=True, title='Model');

# plt.subplot(1,2,2)
# sns.lineplot(x='epoch', y='swap', data=cifar_cnn, label='CNN',color=p[0]); sns.lineplot(x='epoch', y='swap', data=cifar_fcn, label='FCN',color=p[-1])
# plt.title('(b) Feature Swap CIFAR'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(shadow=True, title='Model')
plt.tight_layout(); plt.savefig('Figures/Paper/swap.{save_type}'.format(
    save_type=SAVE_TYPE
))
pbar.update(1)

# # Supplementary Material Figures
# ---
# ### 1. Augmentation Type 
#     * (left: aug_type training, right: un-augmented val acc)
#     * (left: translation val, right: rotation val)
#     * (left: noise val, right: edge noise val)
#     * swap val

# In[19]:


# translation_sm_fig(mnist_cnn, aug_type='rotation', save=True)
# pbar.update(1)
#
# # In[ ]:
#
#
# translation_sm_fig(mnist_cnn, aug_type='noise', save=True)
# pbar.update(1)
#
# # In[ ]:
#
#
# translation_sm_fig(mnist_cnn, aug_type='edge_noise', save=True)
# pbar.update(1)
#
# # In[ ]:
#
#
# translation_sm_fig(mnist_fcn, aug_type='rotation', save=True)
# pbar.update(1)
#
# # In[ ]:
#
#
# translation_sm_fig(mnist_fcn, aug_type='noise', save=True)
# pbar.update(1)
#
# # In[20]:
#
#
# translation_sm_fig(mnist_fcn, aug_type='edge_noise', save=True)
# pbar.update(1)

# In[ ]:


translation_sm_fig(cifar_cnn, aug_type='rotation', cifar=True, save=True)
pbar.update(1)

# In[ ]:


translation_sm_fig(cifar_cnn, aug_type='noise', cifar=True, save=True)
pbar.update(1)

# In[21]:


translation_sm_fig(cifar_cnn, aug_type='edge_noise', cifar=True, save=True)
pbar.update(1)

# In[ ]:


translation_sm_fig(cifar_fcn, aug_type='rotation', cifar=True, save=True)
pbar.update(1)

# In[ ]:


translation_sm_fig(cifar_fcn, aug_type='noise', cifar=True, save=True)
pbar.update(1)

# In[22]:


translation_sm_fig(cifar_fcn, aug_type='edge_noise', cifar=True, save=True)
pbar.update(1)

# ### 2. Variable Connection Pattersn

# In[23]:


# translation_sm_fig(
#     mnist_fcn_vcp[mnist_fcn_vcp['vcp'] == 0.1].reset_index(),
#     aug_type='vcp'
# )
# pbar.update(1)

# In[24]:


translation_sm_fig(
    cifar_fcn_vcp[cifar_fcn_vcp['vcp'] == 0.1].reset_index(), 
    aug_type='vcp',
    cifar=True
)
pbar.update(1)