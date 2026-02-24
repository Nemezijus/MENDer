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


export default function AlgoParamsSwitch({ algo, m, set, sub, enums }) {
  if (!algo) return null;
  switch (algo) {
    case 'agglo':
      return <AggloSection m={m} set={set} sub={sub} enums={enums} />;
    case 'bayridge':
      return <BayridgeSection m={m} set={set} sub={sub} enums={enums} />;
    case 'bgmm':
      return <BgmmSection m={m} set={set} sub={sub} enums={enums} />;
    case 'birch':
      return <BirchSection m={m} set={set} sub={sub} enums={enums} />;
    case 'dbscan':
      return <DbscanSection m={m} set={set} sub={sub} enums={enums} />;
    case 'enet':
      return <EnetSection m={m} set={set} sub={sub} enums={enums} />;
    case 'enetcv':
      return <EnetcvSection m={m} set={set} sub={sub} enums={enums} />;
    case 'extratrees':
      return <ExtratreesSection m={m} set={set} sub={sub} enums={enums} />;
    case 'forest':
      return <ForestSection m={m} set={set} sub={sub} enums={enums} />;
    case 'gmm':
      return <GmmSection m={m} set={set} sub={sub} enums={enums} />;
    case 'gnb':
      return <GnbSection m={m} set={set} sub={sub} enums={enums} />;
    case 'hgb':
      return <HgbSection m={m} set={set} sub={sub} enums={enums} />;
    case 'kmeans':
      return <KmeansSection m={m} set={set} sub={sub} enums={enums} />;
    case 'knn':
      return <KnnSection m={m} set={set} sub={sub} enums={enums} />;
    case 'knnreg':
      return <KnnregSection m={m} set={set} sub={sub} enums={enums} />;
    case 'lasso':
      return <LassoSection m={m} set={set} sub={sub} enums={enums} />;
    case 'lassocv':
      return <LassocvSection m={m} set={set} sub={sub} enums={enums} />;
    case 'linreg':
      return <LinregSection m={m} set={set} sub={sub} enums={enums} />;
    case 'linsvr':
      return <LinsvrSection m={m} set={set} sub={sub} enums={enums} />;
    case 'logreg':
      return <LogregSection m={m} set={set} sub={sub} enums={enums} />;
    case 'meanshift':
      return <MeanshiftSection m={m} set={set} sub={sub} enums={enums} />;
    case 'rfreg':
      return <RfregSection m={m} set={set} sub={sub} enums={enums} />;
    case 'ridge':
      return <RidgeSection m={m} set={set} sub={sub} enums={enums} />;
    case 'ridgecv':
      return <RidgecvSection m={m} set={set} sub={sub} enums={enums} />;
    case 'ridgereg':
      return <RidgeregSection m={m} set={set} sub={sub} enums={enums} />;
    case 'sgd':
      return <SgdSection m={m} set={set} sub={sub} enums={enums} />;
    case 'spectral':
      return <SpectralSection m={m} set={set} sub={sub} enums={enums} />;
    case 'svm':
      return <SvmSection m={m} set={set} sub={sub} enums={enums} />;
    case 'svr':
      return <SvrSection m={m} set={set} sub={sub} enums={enums} />;
    case 'tree':
      return <TreeSection m={m} set={set} sub={sub} enums={enums} />;
    case 'treereg':
      return <TreeregSection m={m} set={set} sub={sub} enums={enums} />;
    default:
      return null;
  }
}
