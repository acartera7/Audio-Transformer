local function change_indent(input_file, output_file, old_indent_size, new_indent_size)
  local old_indent = string.rep(" ", old_indent_size)
  local new_indent = string.rep(" ", new_indent_size)

  local input = io.open(input_file, "r")
  if not input then
    print("Error: Could not open input file.")
    return
  end

  local output = io.open(output_file, "w")
  if not output then
    print("Error: Could not open output file.")
    input:close()
    return
  end

  for line in input:lines() do
    local new_line = line:gsub("^" .. old_indent, new_indent)
    output:write(new_line .. "\n")
  end

  input:close()
  output:close()
end

local function main()
  if #arg < 3 then
    print("Usage: lua change_indent.lua <input_file> <old_indent_size> <new_indent_size>")
    return
  end

  local input_file = arg[1]
  local old_indent_size = tonumber(arg[2])
  local new_indent_size = tonumber(arg[3])
  local output_file = input_file:gsub("%.%w+$", "_new%0")

  if not old_indent_size or not new_indent_size then
    print("Error: Indent sizes must be numbers.")
    return
  end

  change_indent(input_file, output_file, old_indent_size, new_indent_size)
  print("Indentation changed and saved to " .. output_file)
end

main()