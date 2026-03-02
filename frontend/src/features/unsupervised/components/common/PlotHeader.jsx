import { Tooltip, Text, Box } from '@mantine/core';

/**
 * Small shared header used above plot tiles.
 */
export default function PlotHeader({ title, help, minHeight = 34 }) {
  const titleNode = (
    <Box mih={minHeight} className="unsupPlotHeader">
      <Text fw={600} size="md" ta="center">
        {title}
      </Text>
    </Box>
  );

  return help ? (
    <Tooltip label={help} withArrow position="top" multiline maw={360}>
      {titleNode}
    </Tooltip>
  ) : (
    titleNode
  );
}
