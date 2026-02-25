import { Tooltip, Text } from '@mantine/core';

/**
 * Small shared header used above plot tiles.
 */
export default function PlotHeader({ title, help, minHeight = 34 }) {
  const titleNode = (
    <div
      style={{
        minHeight,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
      }}
    >
      <Text fw={600} size="md" ta="center">
        {title}
      </Text>
    </div>
  );

  return help ? (
    <Tooltip label={help} withArrow position="top" multiline maw={360}>
      {titleNode}
    </Tooltip>
  ) : (
    titleNode
  );
}
