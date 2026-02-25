import { Stack, Text } from '@mantine/core';

import { safeMean, safeWeightedMean, computeImbalanceInfo } from '../../utils/classificationMath.js';

import ClassificationMetricHeader from './classificationMetric/ClassificationMetricHeader.jsx';
import ClassificationMetricTable from './classificationMetric/ClassificationMetricTable.jsx';
import { getClassificationMetricTooltips } from './classificationMetric/tooltips.js';

export default function ClassificationMetricResults({ confusion, metricName }) {
  if (!confusion) return null;

  const { overall, macro_avg, weighted_avg, per_class } = confusion;

  if (!Array.isArray(per_class) || per_class.length === 0) {
    return null;
  }

  const fmt = (v) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : '—');

  const fmtRatio = (v) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : '—');

  const supports = per_class.map((c) => c.support ?? 0);
  const totalSupport = supports.reduce((acc, s) => acc + s, 0);

  const nClasses = per_class.length;
  const isMulticlass = nClasses > 2;

  const { ratio: imbalanceRatio, desc: imbalanceDesc } = computeImbalanceInfo(supports);

  // Macro values (precision/recall/f1 from backend if present; otherwise from per_class)
  const macroPrecision = macro_avg?.precision ?? safeMean(per_class.map((c) => c.precision));
  const macroRecall = macro_avg?.recall ?? safeMean(per_class.map((c) => c.recall ?? c.tpr));
  const macroF1 = macro_avg?.f1 ?? safeMean(per_class.map((c) => c.f1));

  const macroTPR = safeMean(per_class.map((c) => c.tpr));
  const macroFPR = safeMean(per_class.map((c) => c.fpr));
  const macroTNR = safeMean(per_class.map((c) => c.tnr));
  const macroFNR = safeMean(per_class.map((c) => c.fnr));
  const macroMCC = macro_avg?.mcc ?? safeMean(per_class.map((c) => c.mcc));

  // Weighted values (precision/recall/f1 from backend if present; otherwise weighted by support)
  const weightedPrecision =
    weighted_avg?.precision ??
    (totalSupport ? safeWeightedMean(per_class.map((c) => c.precision), supports) : null);

  const weightedRecall =
    weighted_avg?.recall ??
    (totalSupport ? safeWeightedMean(per_class.map((c) => c.recall ?? c.tpr), supports) : null);

  const weightedF1 =
    weighted_avg?.f1 ?? (totalSupport ? safeWeightedMean(per_class.map((c) => c.f1), supports) : null);

  const weightedTPR = totalSupport ? safeWeightedMean(per_class.map((c) => c.tpr), supports) : null;
  const weightedFPR = totalSupport ? safeWeightedMean(per_class.map((c) => c.fpr), supports) : null;
  const weightedTNR = totalSupport ? safeWeightedMean(per_class.map((c) => c.tnr), supports) : null;
  const weightedFNR = totalSupport ? safeWeightedMean(per_class.map((c) => c.fnr), supports) : null;

  const weightedMCC =
    weighted_avg?.mcc ?? (totalSupport ? safeWeightedMean(per_class.map((c) => c.mcc), supports) : null);

  const tooltips = getClassificationMetricTooltips({ isMulticlass });

  return (
    <Stack gap="xs" mt={2}>
      <Text fw={500} size="xl" ta="center">
        Summary of metrics
      </Text>

      <ClassificationMetricHeader
        overall={overall}
        imbalanceRatio={imbalanceRatio}
        imbalanceDesc={imbalanceDesc}
        fmt={fmt}
        fmtRatio={fmtRatio}
      />

      <ClassificationMetricTable
        perClass={per_class}
        macro={{
          precision: macroPrecision,
          tpr: macroTPR ?? macroRecall,
          fpr: macroFPR,
          tnr: macroTNR,
          fnr: macroFNR,
          f1: macroF1,
          mcc: macroMCC,
        }}
        weighted={{
          precision: weightedPrecision,
          tpr: weightedTPR ?? weightedRecall,
          fpr: weightedFPR,
          tnr: weightedTNR,
          fnr: weightedFNR,
          f1: weightedF1,
          mcc: weightedMCC,
        }}
        fmt={fmt}
        tooltips={tooltips}
      />
    </Stack>
  );
}
