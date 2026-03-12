import { Card, Stack, Text, Divider } from '@mantine/core';

import DecoderHeader from '../decoder/DecoderHeader.jsx';
import DecoderSummary from '../decoder/DecoderSummary.jsx';
import DecoderClassProbabilitySummary from '../decoder/DecoderClassProbabilitySummary.jsx';
import DecoderExportActions from '../decoder/DecoderExportActions.jsx';
import DecoderPreviewTable from '../decoder/DecoderPreviewTable.jsx';
import DecoderNotes from '../decoder/DecoderNotes.jsx';

import { useDecoderOutputsResultsModel } from '../../hooks/useDecoderOutputsResultsModel.js';

export default function DecoderOutputsResults({ trainResult }) {
  if (!trainResult) return null;

  const vm = useDecoderOutputsResultsModel(trainResult);
  if (!vm.ready) return null;

  return (
    <Card withBorder shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500} size="xl" ta="center">
          Decoder outputs
        </Text>

        <DecoderHeader
          isKfold={vm.isKfold}
          isHoldout={vm.isHoldout}
          nSplits={vm.nSplits}
          isRegression={vm.isRegression}
          hasDecisionScores={vm.hasDecisionScores}
          hasProbabilities={vm.hasProbabilities}
          isVoteShare={vm.isVoteShare}
          previewN={vm.previewN}
          totalN={vm.totalN}
        />

        <Divider />

        <DecoderSummary
          isRegression={vm.isRegression}
          summary={vm.summary}
          regPerfItems={vm.regPerfItems}
          regDataItems={vm.regDataItems}
          dataParamsItems={vm.dataParamsItems}
          lossCalItems={vm.lossCalItems}
          confidenceItems={vm.confidenceItems}
          datasetTitleTip={vm.datasetTitleTip}
          regTitleTip={vm.regTitleTip}
          lossTitleTip={vm.lossTitleTip}
          confTitleTip={vm.confTitleTip}
          showCalibrationBins={vm.showCalibrationBins}
          nonEmptyBins={vm.nonEmptyBins}
        />

        {vm.showClassSummary ? (
          <>
            <Divider />
            <DecoderClassProbabilitySummary
              show={vm.showClassSummary}
              classOptions={vm.classOptions}
              selectedClass={vm.selectedClass}
              setSelectedClass={vm.setSelectedClass}
              probStats={vm.probStats}
            />
            <Divider />
          </>
        ) : (
          <Divider />
        )}

        <DecoderExportActions
          onExportPreview={vm.handleExportPreview}
          onExportFull={vm.handleExportFull}
          isExportingFull={vm.isExportingFull}
          canExportFull={Boolean(trainResult?.artifact?.uid)}
        />

        <DecoderPreviewTable preview={vm.preview} columns={vm.columns} />

        <DecoderNotes notes={vm.decoderNotes} />
      </Stack>
    </Card>
  );
}
