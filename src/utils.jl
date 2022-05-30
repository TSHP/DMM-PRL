function string_to_vec(string)
    if string[1:4] == "Any[" && last(string) == ']'
        vec = []
        tmp = string[5:length(string)-1]
        entries = split(tmp, ',')
        for entry in entries
            if isempty(entry)
                return []
            else
                append!(vec, parse(Float64, entry))
            end
        end
    else
        throw("Unexpected string, must have form with 'Any[' ... ']' ")
    end
    return vec
end