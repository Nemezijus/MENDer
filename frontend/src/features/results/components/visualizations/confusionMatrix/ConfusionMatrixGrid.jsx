import { Box, Text } from '@mantine/core';

export default function ConfusionMatrixGrid({ matrix, labels }) {
  const size = labels.length;

  const MATRIX_SIZE = 320; // px – keeps matrix square
  const Y_AXIS_COL_WIDTH = 14; // "True" column – kept very narrow
  const SUM_BAR_THICKNESS = 40; // height of top sums row == width of right sums column

  // ------------- Main matrix color scale (white → blue, hide zeros) -------------
  const flat = matrix.flat();
  const maxValue = flat.length ? Math.max(...flat.map((v) => Math.abs(v))) || 1 : 1;

  const getCellStyles = (v) => {
    if (!v) {
      // Zero: pure white, no value shown
      return {
        backgroundColor: '#ffffff',
        color: 'transparent',
      };
    }

    const t = maxValue > 0 ? v / maxValue : 0; // 0..1
    const lightness = 100 - 55 * t; // 100% -> 45%
    const bg = `hsl(210, 80%, ${lightness}%)`;
    const color = t > 0.5 ? '#ffffff' : '#000000';

    return {
      backgroundColor: bg,
      color,
    };
  };

  // ------------- Sums (shared scale for row & column sums) ----------------------

  // Column sums (sum down each predicted column)
  const colSums = Array.from({ length: size }, (_, j) => matrix.reduce((acc, row) => acc + (row[j] ?? 0), 0));

  // Row sums (sum across each true row)
  const rowSums = matrix.map((row) => row.reduce((acc, v) => acc + (v ?? 0), 0));

  const allSums = [...colSums, ...rowSums];
  const maxSum = allSums.length ? Math.max(...allSums.map((v) => Math.abs(v))) || 1 : 1;

  const getSumCellStyles = (v) => {
    if (!v) {
      return {
        backgroundColor: '#ffffff',
        color: '#000000',
      };
    }

    const t = maxSum > 0 ? v / maxSum : 0; // 0..1
    const lightness = 100 - 20 * t;
    const bg = `hsl(200, 20%, ${lightness}%)`;
    const color = '#000000';

    return {
      backgroundColor: bg,
      color,
    };
  };

  // Keep colors static (no Mantine CSS-variable dependency)
  const borderColor = '#d1d5db'; // gray-300
  const innerBorderColor = '#e5e7eb'; // gray-200

  return (
    <Box className="cmOuter" my="lg">
      <Box
        className="cmGrid"
        style={{
          gridTemplateColumns: `${Y_AXIS_COL_WIDTH}px max-content ${MATRIX_SIZE}px max-content`,
          gridTemplateRows: 'auto auto auto auto',
        }}
      >
        <Box className="cmPredictedTitle">
          <Text size="md" fw={600}>
            Predicted
          </Text>
        </Box>

        <Box
          className="cmTrueAxis"
          style={{ height: MATRIX_SIZE }}
        >
          <Text
            size="md"
            fw={600}
            className="cmTrueAxisLabel"
          >
            True
          </Text>
        </Box>

        <Box
          className="cmColSums cmBorderBox"
          style={{
            width: MATRIX_SIZE,
            height: SUM_BAR_THICKNESS,
            gridTemplateColumns: `repeat(${size}, 1fr)`,
            borderColor,
          }}
        >
          {colSums.map((sum, j) => {
            const styles = getSumCellStyles(sum);
            const isLast = j === size - 1;
            return (
              <Box
                key={`colsum-${j}`}
                style={{
                  ...styles,
                  borderRight: isLast ? 'none' : `1px solid ${innerBorderColor}`,
                }}
                className="cmSumCell"
              >
                {sum}
              </Box>
            );
          })}
        </Box>

        <Box
          className="cmYLabels"
          style={{
            height: MATRIX_SIZE,
            gridTemplateRows: `repeat(${size}, 1fr)`,
          }}
        >
          {labels.map((lbl, i) => (
            <Box
              key={`y-${i}`}
              className="cmYLabelCell"
            >
              <Text size="xs">{String(lbl)}</Text>
            </Box>
          ))}
        </Box>

        <Box
          className="cmMatrix cmBorderBox"
          style={{
            width: MATRIX_SIZE,
            height: MATRIX_SIZE,
            gridTemplateColumns: `repeat(${size}, 1fr)`,
            gridTemplateRows: `repeat(${size}, 1fr)`,
            borderColor,
          }}
        >
          {matrix.map((row, i) =>
            row.map((v, j) => {
              const styles = getCellStyles(v);
              return (
                <Box
                  key={`${i}-${j}`}
                  className="cmMatrixCell"
                  style={styles}
                >
                  {v ? v : ''}
                </Box>
              );
            }),
          )}
        </Box>

        <Box
          className="cmRowSums cmBorderBox"
          style={{
            height: MATRIX_SIZE,
            width: SUM_BAR_THICKNESS,
            gridTemplateRows: `repeat(${size}, 1fr)`,
            borderColor,
          }}
        >
          {rowSums.map((sum, i) => {
            const styles = getSumCellStyles(sum);
            const isLast = i === size - 1;
            return (
              <Box
                key={`rowsum-${i}`}
                style={{
                  ...styles,
                  borderBottom: isLast ? 'none' : `1px solid ${innerBorderColor}`,
                }}
                className="cmSumCell"
              >
                {sum}
              </Box>
            );
          })}
        </Box>

        <Box
          className="cmXLabels"
          style={{
            width: MATRIX_SIZE,
            gridTemplateColumns: `repeat(${size}, 1fr)`,
          }}
        >
          {labels.map((lbl, j) => (
            <Box
              key={`x-${j}`}
              className="cmXLabelCell"
            >
              <Text size="xs">{String(lbl)}</Text>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
}
