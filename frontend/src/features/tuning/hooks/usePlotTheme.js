import { useMantineTheme } from '@mantine/core';

/**
 * Small helper to keep Plotly theme tokens consistent across tuning result panels.
 */
export default function usePlotTheme() {
  const theme = useMantineTheme();
  const isDark = theme.colorScheme === 'dark';

  return {
    isDark,
    textColor: isDark ? theme.colors.gray[2] : theme.black,
    gridColor: isDark ? theme.colors.dark[4] : '#e0e0e0',
    axisColor: isDark ? theme.colors.dark[2] : '#222',
  };
}
