import React from 'react';
import AggloSection from './sections/agglo.jsx';
import BayridgeSection from './sections/bayridge.jsx';
import BgmmSection from './sections/bgmm.jsx';
import BirchSection from './sections/birch.jsx';
import DbscanSection from './sections/dbscan.jsx';
import EnetSection from './sections/enet.jsx';
import EnetcvSection from './sections/enetcv.jsx';
import ExtratreesSection from './sections/extratrees.jsx';
import ForestSection from './sections/forest.jsx';
import GmmSection from './sections/gmm.jsx';
import GnbSection from './sections/gnb.jsx';
import HgbSection from './sections/hgb.jsx';
import KmeansSection from './sections/kmeans.jsx';
import KnnSection from './sections/knn.jsx';
import KnnregSection from './sections/knnreg.jsx';
import LassoSection from './sections/lasso.jsx';
import LassocvSection from './sections/lassocv.jsx';
import LinregSection from './sections/linreg.jsx';
import LinsvrSection from './sections/linsvr.jsx';
import LogregSection from './sections/logreg.jsx';
import MeanshiftSection from './sections/meanshift.jsx';
import RfregSection from './sections/rfreg.jsx';
import RidgeSection from './sections/ridge.jsx';
import RidgecvSection from './sections/ridgecv.jsx';
import RidgeregSection from './sections/ridgereg.jsx';
import SgdSection from './sections/sgd.jsx';
import SpectralSection from './sections/spectral.jsx';
import SvmSection from './sections/svm.jsx';
import SvrSection from './sections/svr.jsx';
import TreeSection from './sections/tree.jsx';
import TreeregSection from './sections/treereg.jsx';


export default function AlgoParamsSwitch({ algo, m, set, sub, enums, d }) {
  if (!algo) return null;
  switch (algo) {
    case 'agglo':
      return <AggloSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'bayridge':
      return <BayridgeSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'bgmm':
      return <BgmmSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'birch':
      return <BirchSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'dbscan':
      return <DbscanSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'enet':
      return <EnetSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'enetcv':
      return <EnetcvSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'extratrees':
      return <ExtratreesSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'forest':
      return <ForestSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'gmm':
      return <GmmSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'gnb':
      return <GnbSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'hgb':
      return <HgbSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'kmeans':
      return <KmeansSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'knn':
      return <KnnSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'knnreg':
      return <KnnregSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'lasso':
      return <LassoSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'lassocv':
      return <LassocvSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'linreg':
      return <LinregSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'linsvr':
      return <LinsvrSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'logreg':
      return <LogregSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'meanshift':
      return <MeanshiftSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'rfreg':
      return <RfregSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'ridge':
      return <RidgeSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'ridgecv':
      return <RidgecvSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'ridgereg':
      return <RidgeregSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'sgd':
      return <SgdSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'spectral':
      return <SpectralSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'svm':
      return <SvmSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'svr':
      return <SvrSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'tree':
      return <TreeSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    case 'treereg':
      return <TreeregSection m={m} set={set} sub={sub} enums={enums} d={d} />;
    default:
      return null;
  }
}
