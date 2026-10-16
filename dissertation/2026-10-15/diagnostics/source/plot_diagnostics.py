"""Standalone aggregate-only diagnostic figure, with no private row data."""
from pathlib import Path
import csv,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
rows=list(csv.DictReader((WORK/'permutation-summary.csv').open()))
old=list(csv.DictReader((WORK/'original-controls-summary.csv').open()))
summary=json.loads((WORK/'diagnostic-summary.json').read_text())
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(13.2,4.5),gridspec_kw={'width_ratios':[1.2,1,1]})
ax=axes[0]
x=[int(r['seed']) for r in rows]
ax.scatter(x,[float(r['auc']) for r in rows],s=14,color='#215b82',label='True validation labels',zorder=3)
ax.scatter(x,[float(r['randomized_validation_auc']) for r in rows],s=12,color='#bf5a32',marker='x',label='Shuffled validation labels',zorder=4)
ax.axhline(.5,color='gray',lw=.8,ls='--')
ax.axhline(summary['original_max_auc'],color='#666666',lw=.8,ls=':',label='Original maximum AUC')
ax.set(xlabel='Training-label permutation seed',ylabel='Validation ROC AUC',ylim=(0,1),title='A. All 99 scheduled permutations')
fig.legend(*ax.get_legend_handles_labels(), fontsize=8,loc='upper left',bbox_to_anchor=(.055,.94),ncol=3,frameon=False)
ax=axes[1]
ax.scatter([float(r['is_https_coefficient']) for r in rows],[float(r['auc']) for r in rows],s=16,color='#215b82')
ax.axhline(.5,color='gray',lw=.8,ls='--');ax.axvline(0,color='gray',lw=.8,ls='--')
ax.set(xlabel='Coefficient of standardized is_https',ylabel='Validation ROC AUC',ylim=(0,1),title='B. Fitted direction and ranking')
ax.text(.03,.04,f"Spearman r = {summary['auc_https_coefficient_spearman']:.3f}",transform=ax.transAxes,fontsize=9)
ax=axes[2]
for r in old:
    ax.plot([0,1],[float(r['original_auc']),float(r['without_https_auc'])],lw=1.2,marker='o',ms=4,label=f"Seed {r['seed']}")
ax.axhline(.5,color='gray',lw=.8,ls='--')
ax.set(xticks=[0,1],xticklabels=['Original\nscore','HTTPS term\nremoved'],ylabel='Validation ROC AUC',ylim=(.35,.75),title='C. Five fixed-score interventions')
ax.legend(fontsize=8,frameon=False,ncol=2,loc='upper right')
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.subplots_adjust(left=.055,right=.99,top=.84,bottom=.27,wspace=.33)
fig.suptitle('Why shuffled training labels can leave substantial validation ranking',fontsize=15,y=.98)
fig.text(.055,.09,'One fixed training/validation partition. All fits retained. Term removal changes the saved logit without refitting; correlated predictors remain.',fontsize=9)
fig.text(.055,.045,'These are conditional sensitivity results, not independent populations, a leakage p-value, or detection performance at a selected threshold.',fontsize=9)
fig.savefig(WORK/'shuffled-label-mechanism.png',dpi=220,bbox_inches='tight',pad_inches=.12)
fig.savefig(WORK/'shuffled-label-mechanism.pdf',bbox_inches='tight',pad_inches=.12)
