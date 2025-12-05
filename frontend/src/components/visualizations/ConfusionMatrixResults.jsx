import { Stack, Text, Box } from '@mantine/core';

export default function ConfusionMatrixResults({ confusion }) {
  if (!confusion || !confusion.matrix || !confusion.labels) {
    return null;
  }

  const { matrix, labels } = confusion;
  const size = labels.length;

  const MATRIX_SIZE = 320;        // px – keeps matrix square
  const Y_AXIS_COL_WIDTH = 14;    // "True" column – kept very narrow
  const SUM_BAR_THICKNESS = 40;   // height of top sums row == width of right sums column

  // ------------- Main matrix color scale (white → blue, hide zeros) -------------
  const flat = matrix.flat();
  const maxValue =
    flat.length ? Math.max(...flat.map((v) => Math.abs(v))) || 1 : 1;

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
  const colSums = Array.from({ length: size }, (_, j) =>
    matrix.reduce((acc, row) => acc + (row[j] ?? 0), 0),
  );

  // Row sums (sum across each true row)
  const rowSums = matrix.map((row) =>
    row.reduce((acc, v) => acc + (v ?? 0), 0),
  );

  const allSums = [...colSums, ...rowSums];
  const maxSum = allSums.length
    ? Math.max(...allSums.map((v) => Math.abs(v))) || 1
    : 1;

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

  const borderColor = 'var(--mantine-color-gray-4)';
  const innerBorderColor = 'var(--mantine-color-gray-3)';

  return (
    <Stack gap="xs">
      <Text fw={500} size="xl" align="center">
        Confusion matrix
      </Text>

      {/* Center everything in parent */}
      <Box
        style={{
          display: 'flex',
          justifyContent: 'center',
          width: '100%',
        }}
        my="lg"
      >
        {/* Grid:
            cols: [y-axis label] [row labels] [matrix] [row sums]
            rows: [predicted label] [column sums] [matrix row] [x labels] */}
        <Box
          style={{
            display: 'grid',
            gridTemplateColumns: `${Y_AXIS_COL_WIDTH}px max-content ${MATRIX_SIZE}px max-content`,
            gridTemplateRows: 'auto auto auto auto',
            columnGap: 4,
            rowGap: 6,
            alignItems: 'center',
          }}
        >
          {/* Predicted axis label – above matrix only */}
          <Box
            style={{
              gridColumn: '3 / 4',
              gridRow: 1,
              display: 'flex',
              justifyContent: 'center',
            }}
          >
            <Text size="md" fw={600}>
              Predicted
            </Text>
          </Box>

          {/* True axis label – aligned to matrix vertically */}
          <Box
            style={{
              gridColumn: 1,
              gridRow: 3,
              height: MATRIX_SIZE,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Text
              size="md"
              fw={600}
              style={{
                writingMode: 'vertical-rl',
                transform: 'rotate(180deg)',
              }}
            >
              True
            </Text>
          </Box>

          {/* Column sums row – table-like strip above matrix */}
          <Box
            style={{
              gridColumn: 3,
              gridRow: 2,
              width: MATRIX_SIZE,
              height: SUM_BAR_THICKNESS,
              display: 'grid',
              gridTemplateColumns: `repeat(${size}, 1fr)`,
              border: `1px solid ${borderColor}`,
              borderRadius: 0,
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
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 13,
                    fontWeight: 600,
                    borderRight: isLast ? 'none' : `1px solid ${innerBorderColor}`,
                  }}
                >
                  {sum}
                </Box>
              );
            })}
          </Box>

          {/* Row labels (True classes) */}
          <Box
            style={{
              gridColumn: 2,
              gridRow: 3,
              height: MATRIX_SIZE,
              display: 'grid',
              gridTemplateRows: `repeat(${size}, 1fr)`,
              alignItems: 'center',
              paddingRight: 4,
            }}
          >
            {labels.map((lbl, i) => (
              <Box
                key={`y-${i}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'flex-end',
                }}
              >
                <Text size="xs">{String(lbl)}</Text>
              </Box>
            ))}
          </Box>

          {/* Confusion matrix grid */}
          <Box
            style={{
              gridColumn: 3,
              gridRow: 3,
              width: MATRIX_SIZE,
              height: MATRIX_SIZE,
              display: 'grid',
              gridTemplateColumns: `repeat(${size}, 1fr)`,
              gridTemplateRows: `repeat(${size}, 1fr)`,
              border: `1px solid ${borderColor}`,
              borderRadius: 0,
            }}
          >
            {matrix.map((row, i) =>
              row.map((v, j) => {
                const styles = getCellStyles(v);
                return (
                  <Box
                    key={`${i}-${j}`}
                    style={{
                      ...styles,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 12,
                      fontWeight: 500,
                    }}
                  >
                    {v ? v : ''}
                  </Box>
                );
              }),
            )}
          </Box>

          {/* Row sums column – table-like strip to the right of matrix */}
          <Box
            style={{
              gridColumn: 4,
              gridRow: 3,
              height: MATRIX_SIZE,
              width: SUM_BAR_THICKNESS,
              display: 'grid',
              gridTemplateRows: `repeat(${size}, 1fr)`,
              border: `1px solid ${borderColor}`,
              borderRadius: 0,
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
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 13,
                    fontWeight: 600,
                    borderBottom: isLast
                      ? 'none'
                      : `1px solid ${innerBorderColor}`,
                  }}
                >
                  {sum}
                </Box>
              );
            })}
          </Box>

          {/* X-axis labels – aligned under matrix columns */}
          <Box
            style={{
              gridColumn: 3,
              gridRow: 4,
              width: MATRIX_SIZE,
              display: 'grid',
              gridTemplateColumns: `repeat(${size}, 1fr)`,
            }}
          >
            {labels.map((lbl, j) => (
              <Box
                key={`x-${j}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Text size="xs">{String(lbl)}</Text>
              </Box>
            ))}
          </Box>
        </Box>
      </Box>
    </Stack>
  );
}
