import { Stack, Text, TextInput, Button, FileButton } from '@mantine/core';

import {
  displayLocalFilePath,
  shortHashFromSavedName,
} from '../../utils/fileDisplay.js';

function StoredAsLine({ uploadInfo }) {
  if (!uploadInfo?.saved_name) return null;
  const sh = shortHashFromSavedName(uploadInfo.saved_name, 7);
  return (
    <Text size="xs" c="dimmed">
      Stored as: [{sh}] {uploadInfo.saved_name}
    </Text>
  );
}

/**
 * Controlled input for a backend path (typed) or local file picker.
 */
export default function FilePathInput({
  label,
  placeholder = 'Paste file path',
  acceptExts,
  value,
  onTextChange,
  onBrowseFile,
  uploadInfo,
}) {
  return (
    <Stack gap={4}>
      <TextInput
        label={label}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onTextChange?.(e.currentTarget.value)}
        rightSectionWidth={86}
        rightSection={
          <FileButton
            onChange={(file) => {
              onBrowseFile?.(file);
              if (file) onTextChange?.(displayLocalFilePath(file));
            }}
            accept={acceptExts}
          >
            {(props) => (
              <Button {...props} size="xs" variant="light">
                Browse
              </Button>
            )}
          </FileButton>
        }
      />
      <StoredAsLine uploadInfo={uploadInfo} />
    </Stack>
  );
}
