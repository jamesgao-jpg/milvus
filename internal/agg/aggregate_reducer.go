package agg

import (
	"context"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	typeutil2 "github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/segcorepb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type GroupAggReducer struct {
	groupByFieldIds  []int64
	aggregates       []*planpb.Aggregate
	hashValsMap      map[uint64]*Bucket
	groupLimit       int64
	schema           *schemapb.CollectionSchema
	firstFieldData   []*schemapb.FieldData
	totalGroupCount  int64
	allRetrieveCount int64
}

func NewGroupAggReducer(groupByFieldIds []int64, aggregates []*planpb.Aggregate, groupLimit int64, schema *schemapb.CollectionSchema) *GroupAggReducer {
	return &GroupAggReducer{
		groupByFieldIds: groupByFieldIds,
		aggregates:      aggregates,
		hashValsMap:     make(map[uint64]*Bucket), // Initialize hashValsMap
		groupLimit:      groupLimit,
		schema:          schema,
	}
}

type AggregationResult struct {
	fieldDatas       []*schemapb.FieldData
	allRetrieveCount int64
}

func NewAggregationResult(fieldDatas []*schemapb.FieldData, allRetrieveCount int64) *AggregationResult {
	if fieldDatas == nil {
		fieldDatas = make([]*schemapb.FieldData, 0)
	}
	return &AggregationResult{
		fieldDatas:       fieldDatas,
		allRetrieveCount: allRetrieveCount,
	}
}

// GetFieldDatas returns the fieldDatas slice
func (ar *AggregationResult) GetFieldDatas() []*schemapb.FieldData {
	return ar.fieldDatas
}

func (ar *AggregationResult) GetAllRetrieveCount() int64 {
	return ar.allRetrieveCount
}

func (reducer *GroupAggReducer) EmptyAggResult() (*AggregationResult, error) {
	helper, err := typeutil.CreateSchemaHelper(reducer.schema)
	if err != nil {
		return nil, err
	}
	ret := NewAggregationResult(nil, 0)
	appendEmptyField := func(fieldId int64) error {
		field, err := helper.GetFieldFromID(fieldId)
		if err != nil {
			return err
		}
		emptyFieldData, err := typeutil.GenEmptyFieldData(field)
		if err != nil {
			return err
		}
		ret.fieldDatas = append(ret.fieldDatas, emptyFieldData)
		return nil
	}

	for _, grpFid := range reducer.groupByFieldIds {
		err := appendEmptyField(grpFid)
		if err != nil {
			return nil, err
		}
	}
	for _, agg := range reducer.aggregates {
		if agg.GetOp() == planpb.AggregateOp_count {
			countField := genEmptyLongFieldData(schemapb.DataType_Int64, []int64{0})
			ret.fieldDatas = append(ret.fieldDatas, countField)
		} else {
			field, err := helper.GetFieldFromID(agg.GetFieldId())
			if err != nil {
				return nil, merr.Wrapf(err, "failed to get field schema for aggregate fieldID %d", agg.GetFieldId())
			}
			resultType, err := getAggregateResultType(agg.GetOp(), field.GetDataType())
			if err != nil {
				return nil, merr.Wrapf(err, "failed to get result type for aggregate fieldID %d", agg.GetFieldId())
			}
			emptyFieldData, err := genEmptyFieldDataByType(resultType)
			if err != nil {
				return nil, merr.Wrapf(err, "failed to generate empty field data for result type %s", resultType.String())
			}
			ret.fieldDatas = append(ret.fieldDatas, emptyFieldData)
		}
	}
	return ret, nil
}

func genEmptyLongFieldData(dataType schemapb.DataType, data []int64) *schemapb.FieldData {
	return &schemapb.FieldData{
		Type: dataType,
		Field: &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: data}},
			},
		},
	}
}

// genEmptyFieldDataByType generates empty field data based on the data type
func genEmptyFieldDataByType(dataType schemapb.DataType) (*schemapb.FieldData, error) {
	switch dataType {
	case schemapb.DataType_Int64:
		return genEmptyLongFieldData(dataType, []int64{0}), nil
	case schemapb.DataType_Double:
		return &schemapb.FieldData{
			Type: dataType,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_DoubleData{DoubleData: &schemapb.DoubleArray{Data: []float64{0}}},
				},
			},
		}, nil
	case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32:
		return &schemapb.FieldData{
			Type: dataType,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_IntData{IntData: &schemapb.IntArray{Data: []int32{0}}},
				},
			},
		}, nil
	case schemapb.DataType_Float:
		return &schemapb.FieldData{
			Type: dataType,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_FloatData{FloatData: &schemapb.FloatArray{Data: []float32{0}}},
				},
			},
		}, nil
	case schemapb.DataType_VarChar, schemapb.DataType_String, schemapb.DataType_Text:
		return &schemapb.FieldData{
			Type: dataType,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_StringData{StringData: &schemapb.StringArray{Data: []string{}}},
				},
			},
		}, nil
	case schemapb.DataType_Timestamptz:
		return genEmptyLongFieldData(dataType, []int64{0}), nil
	default:
		// For other types, try to use the original field's GenEmptyFieldData
		return nil, merr.WrapErrParameterInvalidMsg("unsupported data type for aggregate result: %s", dataType.String())
	}
}

// getAggregateResultType returns the expected result type for an aggregate operation
// based on the aggregate operator type and the input field type.
func getAggregateResultType(op planpb.AggregateOp, inputType schemapb.DataType) (schemapb.DataType, error) {
	switch op {
	case planpb.AggregateOp_count:
		// count aggregation always returns Int64
		return schemapb.DataType_Int64, nil
	case planpb.AggregateOp_avg:
		// avg aggregation always returns Double
		return schemapb.DataType_Double, nil
	case planpb.AggregateOp_min, planpb.AggregateOp_max:
		// min/max keep the original field type
		return inputType, nil
	case planpb.AggregateOp_sum:
		// sum returns Int64 for integer types, Double for float types
		switch inputType {
		case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32, schemapb.DataType_Int64:
			return schemapb.DataType_Int64, nil
		case schemapb.DataType_Timestamptz:
			return schemapb.DataType_Timestamptz, nil
		case schemapb.DataType_Float, schemapb.DataType_Double:
			return schemapb.DataType_Double, nil
		default:
			return schemapb.DataType_None, merr.WrapErrParameterInvalidMsg("unsupported input type %s for sum aggregation", inputType.String())
		}
	default:
		return schemapb.DataType_None, merr.WrapErrParameterInvalidMsg("unknown aggregate operator: %d", op)
	}
}

// validateAggregationResults validates the input AggregationResult slice
// It checks:
// 1. Each result's fieldDatas length equals numGroupingKeys + numAggs
// 2. No nil fieldData in any result
// 3. Each fieldData's Type matches the expected type from schema
func (reducer *GroupAggReducer) validateAggregationResults(results []*AggregationResult) error {
	if reducer.schema == nil {
		return merr.WrapErrParameterInvalidMsg("schema is nil, cannot validate field types")
	}

	helper, err := typeutil.CreateSchemaHelper(reducer.schema)
	if err != nil {
		return merr.Wrap(err, "failed to create schema helper")
	}

	numGroupingKeys := len(reducer.groupByFieldIds)
	numAggs := len(reducer.aggregates)
	expectedColumnCount := numGroupingKeys + numAggs

	// Build expected types for each column
	expectedTypes := make([]schemapb.DataType, 0, expectedColumnCount)

	// Add types for grouping keys
	for _, fieldID := range reducer.groupByFieldIds {
		field, err := helper.GetFieldFromID(fieldID)
		if err != nil {
			return merr.Wrapf(err, "failed to get field schema for groupBy fieldID %d", fieldID)
		}
		expectedTypes = append(expectedTypes, field.GetDataType())
	}

	// Add types for aggregates
	for _, agg := range reducer.aggregates {
		var expectedType schemapb.DataType
		if agg.GetOp() == planpb.AggregateOp_count {
			// count aggregation always returns Int64
			expectedType = schemapb.DataType_Int64
		} else {
			field, err := helper.GetFieldFromID(agg.GetFieldId())
			if err != nil {
				return merr.Wrapf(err, "failed to get field schema for aggregate fieldID %d", agg.GetFieldId())
			}
			expectedType, err = getAggregateResultType(agg.GetOp(), field.GetDataType())
			if err != nil {
				return merr.Wrapf(err, "failed to get aggregate result type for aggregate fieldID %d", agg.GetFieldId())
			}
		}
		expectedTypes = append(expectedTypes, expectedType)
	}

	// Validate each result
	for resultIdx, result := range results {
		if result == nil {
			return merr.WrapErrServiceInternalMsg("result at index %d is nil", resultIdx)
		}

		fieldDatas := result.GetFieldDatas()

		// Check 1: fieldDatas length
		if len(fieldDatas) != expectedColumnCount {
			return merr.WrapErrServiceInternalMsg("result at index %d has fieldDatas length %d, expected %d (numGroupingKeys=%d, numAggs=%d)",
				resultIdx, len(fieldDatas), expectedColumnCount, numGroupingKeys, numAggs)
		}

		// Check 2: no nil fieldData and Check 3: type matching
		for colIdx, fieldData := range fieldDatas {
			if fieldData == nil {
				return merr.WrapErrServiceInternalMsg("result at index %d has nil fieldData at column %d", resultIdx, colIdx)
			}

			expectedType := expectedTypes[colIdx]
			actualType := fieldData.GetType()
			if actualType != expectedType {
				return merr.WrapErrServiceInternalMsg("result at index %d, column %d has type %s, expected %s",
					resultIdx, colIdx, schemapb.DataType_name[int32(actualType)], schemapb.DataType_name[int32(expectedType)])
			}
		}
	}

	return nil
}

func (reducer *GroupAggReducer) Reduce(ctx context.Context, results []*AggregationResult) (*AggregationResult, error) {
	if len(results) == 0 {
		return reducer.EmptyAggResult()
	}
	if len(results) == 1 {
		if err := reducer.validateAggregationResults(results); err != nil {
			return nil, err
		}
		return results[0], nil
	}
	if len(reducer.groupByFieldIds) > 0 {
		if err := reducer.Merge(results); err != nil {
			return nil, err
		}
		return reducer.Result()
	}
	if err := reducer.validateAggregationResults(results); err != nil {
		return nil, err
	}

	aggs := make([]AggregateBase, len(reducer.aggregates))
	for idx, aggPb := range reducer.aggregates {
		agg, err := FromPB(aggPb)
		if err != nil {
			return nil, err
		}
		aggs[idx] = agg
	}

	firstFieldData := results[0].GetFieldDatas()
	outputColumnCount := len(firstFieldData)
	accumulators := make([]FieldAccessor, len(reducer.aggregates))
	for idx, fieldData := range firstFieldData {
		accessor, err := NewFieldAccessor(fieldData.GetType())
		if err != nil {
			return nil, err
		}
		accumulators[idx] = accessor
	}

	reducedResult := NewAggregationResult(nil, 0)
	reducedResult.fieldDatas = typeutil.PrepareResultFieldData(firstFieldData, 1)
	rows := make([]*Row, len(results))
	for idx, result := range results {
		reducedResult.allRetrieveCount += result.GetAllRetrieveCount()
		fieldValues := make([]*FieldValue, outputColumnCount)
		for col := 0; col < outputColumnCount; col++ {
			fieldData := result.GetFieldDatas()[col]
			accumulators[col].SetVals(fieldData)
			if accumulators[col].IsNullAt(0) {
				fieldValues[col] = NewNullFieldValue()
			} else {
				fieldValues[col] = NewFieldValue(accumulators[col].ValAt(0))
			}
		}
		rows[idx] = NewRow(fieldValues)
	}
	for row := 1; row < len(rows); row++ {
		for col := 0; col < outputColumnCount; col++ {
			rows[0].UpdateFieldValue(rows[row], col, aggs[col])
		}
	}
	AssembleSingleRow(outputColumnCount, rows[0], reducedResult.fieldDatas)
	return reducedResult, nil
}

// Merge accumulates grouped partial results without assembling the output.
func (reducer *GroupAggReducer) Merge(results []*AggregationResult) error {
	if len(reducer.groupByFieldIds) == 0 {
		return merr.WrapErrServiceInternalMsg("incremental group aggregation requires at least one Group By field")
	}
	if len(results) == 0 {
		return nil
	}
	if err := reducer.validateAggregationResults(results); err != nil {
		return err
	}

	aggs := make([]AggregateBase, len(reducer.aggregates))
	for idx, aggPb := range reducer.aggregates {
		agg, err := FromPB(aggPb)
		if err != nil {
			return err
		}
		aggs[idx] = agg
	}

	numGroupingKeys := len(reducer.groupByFieldIds)
	firstFieldData := results[0].GetFieldDatas()
	outputColumnCount := len(firstFieldData)
	if reducer.firstFieldData == nil {
		reducer.firstFieldData = typeutil.PrepareResultFieldData(firstFieldData, 0)
	}
	hashers := make([]FieldAccessor, numGroupingKeys)
	accumulators := make([]FieldAccessor, len(reducer.aggregates))
	for idx, fieldData := range firstFieldData {
		accessor, err := NewFieldAccessor(fieldData.GetType())
		if err != nil {
			return err
		}
		if idx < numGroupingKeys {
			hashers[idx] = accessor
		} else {
			accumulators[idx-numGroupingKeys] = accessor
		}
	}

	maxGroupByGroups := paramtable.Get().CommonCfg.GroupByMaxGroups.GetAsInt64()
	for _, result := range results {
		reducer.allRetrieveCount += result.GetAllRetrieveCount()
		fieldDatas := result.GetFieldDatas()
		if outputColumnCount != len(fieldDatas) {
			return merr.WrapErrServiceInternalMsg("retrieved results from different segments have different size of columns")
		}
		if outputColumnCount == 0 {
			return merr.WrapErrServiceInternalMsg("retrieved results have no column data")
		}
		rowCount := -1
		for i := 0; i < outputColumnCount; i++ {
			fieldData := fieldDatas[i]
			if i < numGroupingKeys {
				hashers[i].SetVals(fieldData)
			} else {
				accumulators[i-numGroupingKeys].SetVals(fieldData)
			}
			if rowCount == -1 {
				rowCount = hashers[i].RowCount()
			} else if i < numGroupingKeys {
				if rowCount != hashers[i].RowCount() {
					return merr.WrapErrServiceInternalMsg("field data:%d for different columns have different row count, %d vs %d, wrong state",
						i, rowCount, hashers[i].RowCount())
				}
			} else if rowCount != accumulators[i-numGroupingKeys].RowCount() {
				return merr.WrapErrServiceInternalMsg("field data:%d for different columns have different row count, %d vs %d, wrong state",
					i, rowCount, accumulators[i-numGroupingKeys].RowCount())
			}
		}

		for row := 0; row < rowCount; row++ {
			rowFieldValues := make([]*FieldValue, outputColumnCount)
			var hashVal uint64
			for col := 0; col < outputColumnCount; col++ {
				if col < numGroupingKeys {
					if col > 0 {
						hashVal = typeutil2.HashMix(hashVal, hashers[col].Hash(row))
					} else {
						hashVal = hashers[col].Hash(row)
					}
					if hashers[col].IsNullAt(row) {
						rowFieldValues[col] = NewNullFieldValue()
					} else {
						rowFieldValues[col] = NewFieldValue(hashers[col].ValAt(row))
					}
				} else if accumulators[col-numGroupingKeys].IsNullAt(row) {
					rowFieldValues[col] = NewNullFieldValue()
				} else {
					rowFieldValues[col] = NewFieldValue(accumulators[col-numGroupingKeys].ValAt(row))
				}
			}

			newRow := NewRow(rowFieldValues)
			bucket := reducer.hashValsMap[hashVal]
			if bucket == nil {
				if reducer.groupLimit != -1 && reducer.totalGroupCount >= reducer.groupLimit {
					continue
				}
				bucket = NewBucket()
				bucket.AddRow(newRow)
				reducer.hashValsMap[hashVal] = bucket
				reducer.totalGroupCount++
			} else if rowIndex := bucket.Find(newRow, numGroupingKeys); rowIndex == NONE {
				if reducer.groupLimit != -1 && reducer.totalGroupCount >= reducer.groupLimit {
					continue
				}
				bucket.AddRow(newRow)
				reducer.totalGroupCount++
			} else if err := bucket.Accumulate(newRow, rowIndex, numGroupingKeys, aggs); err != nil {
				return err
			}
			if reducer.totalGroupCount > maxGroupByGroups {
				return merr.WrapErrParameterInvalidMsg("GROUP BY produced too many groups (%d). "+
					"Add filters or increase common.groupBy.maxGroups (current: %d)",
					reducer.totalGroupCount, maxGroupByGroups)
			}
		}
	}
	return nil
}

// Result assembles the accumulated grouped result.
func (reducer *GroupAggReducer) Result() (*AggregationResult, error) {
	if reducer.firstFieldData == nil {
		return reducer.EmptyAggResult()
	}
	reducedResult := NewAggregationResult(nil, reducer.allRetrieveCount)
	reducedResult.fieldDatas = typeutil.PrepareResultFieldData(reducer.firstFieldData, reducer.totalGroupCount)
	for _, bucket := range reducer.hashValsMap {
		if err := AssembleBucket(bucket, reducedResult.GetFieldDatas()); err != nil {
			return nil, err
		}
	}
	return reducedResult, nil
}

func InternalResult2AggResult(results []*internalpb.RetrieveResults) []*AggregationResult {
	aggResults := make([]*AggregationResult, len(results))
	for i := 0; i < len(results); i++ {
		aggResults[i] = NewAggregationResult(results[i].GetFieldsData(), results[i].GetAllRetrieveCount())
	}
	return aggResults
}

func AggResult2internalResult(aggRes *AggregationResult) *internalpb.RetrieveResults {
	return &internalpb.RetrieveResults{FieldsData: aggRes.GetFieldDatas(), AllRetrieveCount: aggRes.GetAllRetrieveCount()}
}

func SegcoreResults2AggResult(results []*segcorepb.RetrieveResults) ([]*AggregationResult, error) {
	aggResults := make([]*AggregationResult, len(results))
	for i := 0; i < len(results); i++ {
		if results[i] == nil {
			return nil, merr.WrapErrServiceInternalMsg("input segcore query results from any sources cannot be nil")
		}
		fieldsData := results[i].GetFieldsData()
		allRetrieveCount := results[i].GetAllRetrieveCount()
		aggResults[i] = NewAggregationResult(fieldsData, allRetrieveCount)
	}
	return aggResults, nil
}

func AggResult2segcoreResult(aggRes *AggregationResult) *segcorepb.RetrieveResults {
	return &segcorepb.RetrieveResults{FieldsData: aggRes.GetFieldDatas(), AllRetrieveCount: aggRes.GetAllRetrieveCount()}
}
