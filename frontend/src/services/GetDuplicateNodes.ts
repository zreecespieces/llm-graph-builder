import { duplicateNodesData } from '../types';
import api from '../API/Index';

export const getDuplicateNodes = async (
  signal: AbortSignal,
  excludeLabels?: string[],
  includeLabels?: string[]
) => {
  const formData = new FormData();
  if (excludeLabels?.length) {
    formData.append('exclude_labels', JSON.stringify(excludeLabels));
  }
  if (includeLabels?.length) {
    formData.append('include_labels', JSON.stringify(includeLabels));
  }
  try {
    const response = await api.post<duplicateNodesData>(`/get_duplicate_nodes`, formData, { signal });
    return response;
  } catch (error) {
    console.log(error);
    throw error;
  }
};
