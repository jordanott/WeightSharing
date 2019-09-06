import matplotlib
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

SAVE_TYPE = 'png'

'''
rm -rf Figures
mkdir Figures
mkdir Figures/Paper
mkdir Figures/SM
'''

titles = {
    'acc': 'Training Accuracy',
    'val_acc': 'Un-Augmented Validation Accuracy',
    'translation': 'Translation Augmented Validation Accuracy',
    'noise': 'Noise Augmented Validation Accuracy',
    'edge_noise': 'Edge Noise Augmented Validation Accuracy',
    'rotation': 'Rotation Augmented Validation Accuracy',
    'swap': 'Swap Quadrants Validation Accuracy',
}

def convert_table(df, eval_type='val_acc', to_latex=False):
    res_table = df.groupby(['model','aug_type','aug'])[eval_type].median().to_frame().reset_index().pivot_table(index='aug',columns=['aug_type', 'model'])
    res_table.columns = res_table.columns.droplevel()
    if to_latex: print res_table.to_latex()
    else: return res_table

def convert_table_vcp(df, eval_type='val_acc', to_latex=False):
    res_table = df.groupby(['vcp','aug'])[eval_type].median().to_frame().reset_index().pivot_table(index='vcp',columns=['aug'])
    res_table.columns = res_table.columns.droplevel()
    if to_latex: print res_table.to_latex()
    else: return res_table

def aug_eval(df, aug_type='translation', y=[0.9, 1.0], eval_type='val_acc'):
    hue = 'vcp' if aug_type == 'vcp' else 'aug'
    df_aug_type = df[df['aug_type'] == aug_type]
    
    palette = sns.color_palette('coolwarm', len(df_aug_type[hue].unique()))

    ax=sns.lineplot(x='epoch', y=eval_type, data=df_aug_type, hue=hue, palette=palette, legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    
    ax.legend(ncol=4,bbox_to_anchor=(.5, .29), loc='upper center', shadow=True).texts[0].set_text('Aug %')
    
def aug_train(cnn, fcn, aug_type='translation', y=[0.9, 1.0], eval_type='val_acc'):
    cnn_aug_type = cnn[cnn['aug_type'] == aug_type]
    fcn_aug_type = fcn[fcn['aug_type'] == aug_type]
    
    cnn_palette = sns.color_palette('coolwarm', len(cnn_aug_type['aug'].unique()))
    fcn_palette = sns.color_palette('coolwarm', len(fcn_aug_type['aug'].unique()))

    plt.subplot(1,2,2); 
    sns.lineplot(x='epoch', y=eval_type, data=cnn_aug_type, hue='aug', palette=cnn_palette, legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs'); plt.ylabel(titles[eval_type])
    
    plt.subplot(1,2,1); 
    sns.lineplot(x='epoch', y=eval_type, data=fcn_aug_type, hue='aug', palette=fcn_palette, legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs'); plt.ylabel(titles[eval_type]); plt.legend(shadow=True); plt.show()

def approx_ws(df, dist_type='cos', aug_type='translation',y=[], save=True, cifar=False):
    df = df[df['aug_type'] ==  aug_type].reset_index()
    palette = sns.color_palette('coolwarm', len(df['aug'].unique()))
    
    plt.clf(); fig=plt.figure(figsize=(20, 6))
    plt.subplot(1,2,1); 
    ax=sns.lineplot(x='epoch', y=dist_type+'_1', data=df, hue='aug',palette=palette,legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs'); plt.title('(a) Filters at Distance 1'); plt.ylabel('Cosine Similarity' if dist_type=='cos' else 'Euclidean Distance')
    ax.legend(ncol=4,bbox_to_anchor=(.5, .29), loc='upper center', shadow=True).texts[0].set_text('Aug %')
    
    plt.subplot(1,2,2); 
    ax=sns.lineplot(x='epoch', y=dist_type+'_4', data=df, hue='aug',palette=palette,legend='full')
    plt.ylim(y[0],y[1]); plt.xlabel('Epochs'); plt.title('(b) Filters at Distance 4'); plt.ylabel('Cosine Similarity' if dist_type=='cos' else 'Euclidean Distance'); 
    ax.legend(ncol=4,bbox_to_anchor=(.5, .29), loc='upper center', shadow=True).texts[0].set_text('Aug %'); 
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save: 
        plt.savefig('Figures/Paper/approximate_ws_{data}.{save_type}'.format(
            data='cifar' if cifar else 'mnist',
            save_type=SAVE_TYPE
        ))
    else: plt.show()
        

def translation_paper_fig(fcn, cnn, cifar=False, save=True):
    data = 'cifar' if cifar else 'mnist'
    
    plt.clf(); fig=plt.figure(figsize=(20, 15))
    plt.subplot(3,2,1); aug_eval(fcn, y=[0,1.05], eval_type='acc'); plt.title('(a) ' + 'Translation Augmented ' + titles['acc'])
    plt.subplot(3,2,2); aug_eval(cnn, y=[0,1.05], eval_type='acc'); plt.title('(b) ' + 'Translation Augmented ' + titles['acc'])
    plt.subplot(3,2,3); aug_eval(fcn, y=[0.2,0.8] if cifar else [0.9,1], eval_type='val_acc'); plt.title('(c) ' + titles['val_acc'])
    plt.subplot(3,2,4); aug_eval(cnn, y=[0.2,0.8] if cifar else [0.9,1], eval_type='val_acc'); plt.title('(d) ' + titles['val_acc'])
    plt.subplot(3,2,5); aug_eval(fcn, y=[0.2,0.8] if cifar else [0,1.05], eval_type='translation'); plt.title('(e) ' + titles['translation'])
    plt.subplot(3,2,6); aug_eval(cnn, y=[0.2,0.8] if cifar else [0,1.05], eval_type='translation'); plt.title('(f) ' + titles['translation'])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.suptitle('Trained On Translation Augmented {data}'.format(
        data=data.upper()
    ))
    
    if save: 
        plt.savefig('Figures/Paper/{data}.{save_type}'.format(
            data=data,
            save_type = SAVE_TYPE
        ))
    else: plt.show()
        
sm_ylims = {
    'mnist':{
        'train': [0.9,1.],
        'val_acc': [0.9,1],
        
        'rotation': {
            'trans': [0., 0.45],
            'rotation':[0.9,1],
            'noise':[.7,1],
            'edge_noise':[0.9,1]
        },
        'noise': {
            'trans': [0., 0.45],
            'rotation':[.7,.9],
            'noise':[0.9,1],
            'edge_noise':[0.9,1]
        },
        'edge_noise': {
            'trans': [0., 0.45],
            'rotation':[.7,.9],
            'noise':[0.9,1],
            'edge_noise':[0.9,1]
        },
        'vcp': {
            'trans': [0., 1],
            'rotation':[0., 1],
            'noise':[0., 1],
            'edge_noise':[0., 1]
        }
    },
    'cifar':{
        'train': [0.2,1],
        'val_acc': [0.2,.8],
        'rotation': {
            'trans': [0.2, .7],
            'rotation':[0.2,.7],
            'noise':[0.2,.7],
            'edge_noise':[0.2,.7]
        },
        'noise': {
            'trans': [0.1, .7],
            'rotation':[0.1,.7],
            'noise':[0.1,.7],
            'edge_noise':[0.1,.7]
        },
        'edge_noise': {
            'trans': [0.2, .7],
            'rotation':[0.2,.7],
            'noise':[0.2,.7],
            'edge_noise':[0.2,.7]
        },
        'vcp': {
            'trans': [0., .8],
            'rotation':[0.,.8],
            'noise':[0.,.8],
            'edge_noise':[0.,.8]
        }
    }
}


def translation_sm_fig(df, aug_type, cifar=False, save=True):
    data = 'cifar' if cifar else 'mnist'
    model = df['model'].unique()[0]
    
    plt.clf(); fig=plt.figure(figsize=(20, 18))
    plt.subplot(3,2,1); aug_eval(df, aug_type=aug_type, eval_type='acc', y=sm_ylims[data]['train']); plt.title('(a) ' + titles['acc'])
    plt.subplot(3,2,2); aug_eval(df, aug_type=aug_type, eval_type='val_acc', y=sm_ylims[data]['val_acc']); plt.title('(b) ' + titles['val_acc'])
    plt.subplot(3,2,3); aug_eval(df, aug_type=aug_type, eval_type='translation', y=sm_ylims[data][aug_type]['trans']); plt.title('(c) ' + titles['translation'])
    plt.subplot(3,2,4); aug_eval(df, aug_type=aug_type, eval_type='rotation', y=sm_ylims[data][aug_type]['rotation']); plt.title('(d) ' + titles['rotation'])
    plt.subplot(3,2,5); aug_eval(df, aug_type=aug_type, eval_type='noise', y=sm_ylims[data][aug_type]['noise']); plt.title('(e) ' + titles['noise'])
    plt.subplot(3,2,6); aug_eval(df, aug_type=aug_type, eval_type='edge_noise', y=sm_ylims[data][aug_type]['edge_noise']); plt.title('(f) ' + titles['edge_noise'])
    #plt.subplot(4,2,7); aug_eval(df, aug_type=aug_type, eval_type='swap', y=[0.2,0.8] if cifar else [0,1.05])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if aug_type == 'vcp':
        suptitle = 'VCP FCN Trained On Translation Augmented ' + data.upper()
    else:
        suptitle = '{model} Trained On {aug_type} {data}'.format(
            model=model.upper(),
            aug_type=titles[aug_type].replace(' Validation Accuracy',''),
            data=data.upper()
        )
    
    fig.suptitle(suptitle)
    
    if save: 
        plt.savefig('Figures/SM/{model}_{aug_type}_{data}.{save_type}'.format(
            model=model,
            aug_type=aug_type,
            data=data,
            save_type=SAVE_TYPE
        ))
    else: plt.show()